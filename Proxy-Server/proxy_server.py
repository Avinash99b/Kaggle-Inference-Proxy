"""
proxy_server.py
================================================================================
Public-facing, OpenAI-compatible API proxy.

This process is the only thing that needs a real public IP / domain / TLS
certificate. It exposes a normal-looking OpenAI API to clients:

    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions
    POST /v1/embeddings          (returns a clean OpenAI-style error - the
                                   llama.cpp GGUF worker does not serve
                                   embeddings in this setup)
    GET  /health
    GET  /dashboard

Behind the scenes it queues each incoming request as a "Job" and waits for a
single privately-connected Kaggle worker to pick it up and return a result.
The worker connects OUTBOUND to this server -- this server never dials into
Kaggle. Two transports are supported for the worker:

    * WebSocket  (ws://.../worker/ws)   <- preferred, low latency, push-based
    * Long-poll  (GET /worker/poll,
                  POST /worker/result)  <- fallback if WS is blocked/unstable

Only one worker is considered "active" at a time. Clients are completely
unaware of any of this; from their point of view it is a normal, stateless
OpenAI-compatible server (no server-side conversation memory is kept).

Run:
    pip install fastapi "uvicorn[standard]" websockets
    python proxy_server.py
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.environ.get("PROXY_CONFIG_PATH", "proxy_server_config.json")

DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "client_api_keys": ["sk-change-me-client-key"],
    "worker_shared_secret": "change-me-worker-secret",
    "max_queue_size": 100,
    "request_timeout_seconds": 300,
    "long_poll_timeout_seconds": 40,
    "job_delivery_ack_timeout_seconds": 15,
    "worker_offline_after_seconds": 90,
    "auth_timestamp_skew_seconds": 300,
    "websocket": {
        "heartbeat_interval_seconds": 20,
        "heartbeat_timeout_seconds": 60,
    },
    "retry": {
        "max_job_retries": 1,
    },
    "logging": {
        "level": "INFO",
        "file": "proxy_server.log",
    },
    "dashboard": {
        "enabled": True,
        "refresh_seconds": 5,
    },
    "cors": {
        "enabled": True,
        "allow_origins": ["*"],
    },
}


def load_or_create_config(path: str, defaults: dict) -> dict:
    """Load JSON config, or write out a default one and tell the user to edit it."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(defaults, f, indent=2)
        print(f"[proxy_server] No config found at '{path}'. A default config "
              f"has been created. Please review/edit it (API keys, secrets, "
              f"ports) before exposing this server publicly.")
        return json.loads(json.dumps(defaults))
    with open(path, "r") as f:
        cfg = json.load(f)
    # Backfill any missing keys from defaults without clobbering user edits.
    merged = json.loads(json.dumps(defaults))
    merged.update(cfg)
    for k, v in defaults.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged[k] = {**v, **cfg[k]}
    return merged


CONFIG = load_or_create_config(CONFIG_PATH, DEFAULT_CONFIG)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

log_level = getattr(logging, CONFIG["logging"].get("level", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["logging"].get("file", "proxy_server.log")),
    ],
)
logger = logging.getLogger("proxy_server")

# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #


def check_client_key(authorization: Optional[str]) -> None:
    """Client -> Proxy auth: simple bearer token, like a real OpenAI API key."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=openai_error(
            "Missing or malformed Authorization header.", "invalid_request_error"))
    token = authorization[len("Bearer "):].strip()
    if token not in CONFIG["client_api_keys"]:
        raise HTTPException(status_code=401, detail=openai_error(
            "Invalid API key.", "invalid_request_error"))


def make_worker_signature(worker_id: str, timestamp: str, secret: str) -> str:
    msg = f"{worker_id}:{timestamp}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_worker_auth(worker_id: str, timestamp: str, signature: str) -> bool:
    """Worker -> Proxy auth: HMAC over worker_id+timestamp with shared secret,
    with a timestamp freshness check to reduce replay risk."""
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return False
    skew = CONFIG.get("auth_timestamp_skew_seconds", 300)
    if abs(time.time() - ts) > skew:
        return False
    expected = make_worker_signature(worker_id, timestamp, CONFIG["worker_shared_secret"])
    return hmac.compare_digest(expected, signature or "")


def openai_error(message: str, err_type: str = "server_error", code: Optional[str] = None,
                  status: Optional[int] = None) -> dict:
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code,
        }
    }


# --------------------------------------------------------------------------- #
# Job / worker state
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    kind: str                      # "chat", "completions"
    payload: dict                  # normalized job payload sent to worker
    created_at: float = field(default_factory=time.time)
    timeout: float = 300.0
    future: "asyncio.Future" = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    delivered: bool = False
    delivered_at: Optional[float] = None
    retries: int = 0
    stream: bool = False


class WorkerState:
    """Tracks the single active worker (whichever transport it is using)."""

    def __init__(self) -> None:
        self.worker_id: Optional[str] = None
        self.transport: Optional[str] = None       # "ws" or "poll"
        self.ws: Optional[WebSocket] = None
        self.last_heartbeat: float = 0.0
        self.models: Dict[str, Any] = {}
        self.status: str = "unknown"                # idle / busy / loading_model / unknown
        self.current_model: Optional[str] = None
        self.lock = asyncio.Lock()

    def is_online(self) -> bool:
        if self.last_heartbeat == 0:
            return False
        return (time.time() - self.last_heartbeat) < CONFIG["worker_offline_after_seconds"]

    def touch(self) -> None:
        self.last_heartbeat = time.time()

    def snapshot(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "transport": self.transport,
            "online": self.is_online(),
            "status": self.status,
            "current_model": self.current_model,
            "last_heartbeat_age_seconds": round(time.time() - self.last_heartbeat, 1)
            if self.last_heartbeat else None,
            "known_models": list(self.models.keys()),
        }


class JobManager:
    """Owns the job queue and the map of in-flight jobs."""

    def __init__(self) -> None:
        self.queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=CONFIG["max_queue_size"])
        self.jobs: Dict[str, Job] = {}

    def qsize(self) -> int:
        return self.queue.qsize()

    async def submit(self, kind: str, payload: dict, stream: bool) -> Job:
        if self.queue.full():
            raise HTTPException(status_code=429, detail=openai_error(
                "Server is overloaded, queue is full. Please retry shortly.",
                "rate_limit_error"))
        job = Job(
            id=str(uuid.uuid4()),
            kind=kind,
            payload=payload,
            timeout=CONFIG["request_timeout_seconds"],
            stream=stream,
        )
        self.jobs[job.id] = job
        await self.queue.put(job.id)
        return job

    def pop_next_for_delivery(self) -> Optional[Job]:
        """Non-blocking pop used by both the WS dispatcher and long-poll."""
        try:
            job_id = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        job = self.jobs.get(job_id)
        if job is None or job.future.done():
            return None
        job.delivered = True
        job.delivered_at = time.time()
        return job

    async def wait_next_for_delivery(self, timeout: float) -> Optional[Job]:
        """Blocking (up to timeout) pop, used by the long-poll endpoint."""
        try:
            job_id = await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        job = self.jobs.get(job_id)
        if job is None or job.future.done():
            return None
        job.delivered = True
        job.delivered_at = time.time()
        return job

    def complete(self, job_id: str, result: Optional[dict], error: Optional[str]) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.future.done():
            return False  # unknown or duplicate delivery -- ignore safely
        if error:
            job.future.set_exception(RuntimeError(error))
        else:
            job.future.set_result(result)
        return True

    async def requeue_or_fail(self, job: Job, reason: str) -> None:
        """Called by the watchdog when a delivered job never got a result."""
        if job.future.done():
            return
        if job.retries < CONFIG["retry"]["max_job_retries"]:
            job.retries += 1
            job.delivered = False
            job.delivered_at = None
            logger.warning("Requeuing job %s after failure: %s", job.id, reason)
            await self.queue.put(job.id)
        else:
            logger.error("Failing job %s permanently: %s", job.id, reason)
            job.future.set_exception(RuntimeError(reason))

    def cleanup(self, max_age: float = 3600) -> None:
        now = time.time()
        stale = [jid for jid, j in self.jobs.items()
                 if j.future.done() and (now - j.created_at) > max_age]
        for jid in stale:
            del self.jobs[jid]


WORKER = WorkerState()
JOBS = JobManager()

# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(title="GGUF-Kaggle OpenAI-Compatible Proxy")

if CONFIG["cors"].get("enabled"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CONFIG["cors"].get("allow_origins", ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --------------------------------------------------------------------------- #
# OpenAI response formatting helpers
# --------------------------------------------------------------------------- #


def format_chat_completion(job_id: str, model: str, result: dict) -> dict:
    return {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.get("text", "")},
            "finish_reason": result.get("finish_reason", "stop"),
        }],
        "usage": result.get("usage", {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }),
    }


def format_text_completion(job_id: str, model: str, result: dict) -> dict:
    return {
        "id": f"cmpl-{job_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": result.get("text", ""),
            "logprobs": None,
            "finish_reason": result.get("finish_reason", "stop"),
        }],
        "usage": result.get("usage", {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }),
    }


async def fake_sse_stream(final_chunk_builder):
    """We do not have true token-by-token streaming across the WS/poll hop to
    Kaggle, so as a compatibility fallback we compute the full result and then
    emit it as a single SSE 'delta' chunk followed by [DONE]. This keeps
    stream=True clients working without them needing special-case code."""
    payload = final_chunk_builder()
    yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


def to_stream_chat_chunk(job_id: str, model: str, result: dict) -> dict:
    return {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": result.get("text", "")},
            "finish_reason": result.get("finish_reason", "stop"),
        }],
    }


def to_stream_text_chunk(job_id: str, model: str, result: dict) -> dict:
    return {
        "id": f"cmpl-{job_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": result.get("text", ""),
            "logprobs": None,
            "finish_reason": result.get("finish_reason", "stop"),
        }],
    }


# --------------------------------------------------------------------------- #
# Shared request handling
# --------------------------------------------------------------------------- #


def normalize_generation_params(body: dict) -> dict:
    """Pull out OpenAI-style generation params with sane defaults."""
    return {
        "temperature": body.get("temperature", 0.8),
        "top_p": body.get("top_p", 0.95),
        "max_tokens": body.get("max_tokens", 512),
        "stop": body.get("stop"),
        "presence_penalty": body.get("presence_penalty", 0.0),
        "frequency_penalty": body.get("frequency_penalty", 0.0),
        "seed": body.get("seed"),
    }


async def run_job_and_wait(kind: str, payload: dict, stream: bool) -> (Job, dict):
    if not WORKER.is_online():
        raise HTTPException(status_code=503, detail=openai_error(
            "No inference worker is currently connected. Please try again shortly.",
            "server_error"))
    job = await JOBS.submit(kind, payload, stream)
    try:
        result = await asyncio.wait_for(job.future, timeout=job.timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=openai_error(
            "Timed out waiting for the inference worker to respond.",
            "server_error"))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=openai_error(
            f"Worker failed to process the request: {e}", "server_error"))
    return job, result


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    check_client_key(authorization)
    body = await request.json()
    model = body.get("model")
    messages = body.get("messages")
    if not model or not messages:
        raise HTTPException(status_code=400, detail=openai_error(
            "'model' and 'messages' are required.", "invalid_request_error"))
    stream = bool(body.get("stream", False))
    payload = {
        "kind": "chat",
        "model": model,
        "messages": messages,
        "params": normalize_generation_params(body),
    }
    job, result = await run_job_and_wait("chat", payload, stream)

    if stream:
        return StreamingResponse(
            fake_sse_stream(lambda: to_stream_chat_chunk(job.id, model, result)),
            media_type="text/event-stream",
        )
    return JSONResponse(format_chat_completion(job.id, model, result))


@app.post("/v1/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    check_client_key(authorization)
    body = await request.json()
    model = body.get("model")
    prompt = body.get("prompt")
    if not model or prompt is None:
        raise HTTPException(status_code=400, detail=openai_error(
            "'model' and 'prompt' are required.", "invalid_request_error"))
    stream = bool(body.get("stream", False))
    payload = {
        "kind": "completion",
        "model": model,
        "prompt": prompt,
        "params": normalize_generation_params(body),
    }
    job, result = await run_job_and_wait("completions", payload, stream)

    if stream:
        return StreamingResponse(
            fake_sse_stream(lambda: to_stream_text_chunk(job.id, model, result)),
            media_type="text/event-stream",
        )
    return JSONResponse(format_text_completion(job.id, model, result))


@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: Optional[str] = Header(None)):
    check_client_key(authorization)
    # This GGUF/llama.cpp worker setup does not serve embeddings.
    return JSONResponse(
        status_code=400,
        content=openai_error(
            "This deployment does not support the embeddings endpoint. "
            "Only chat/completions models are available via this worker.",
            "invalid_request_error",
            code="embeddings_not_supported",
        ),
    )


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    check_client_key(authorization)
    now = int(time.time())
    data = [
        {"id": alias, "object": "model", "created": now, "owned_by": "kaggle-worker"}
        for alias in WORKER.models.keys()
    ]
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    JOBS.cleanup()
    return {
        "status": "ok",
        "worker": WORKER.snapshot(),
        "queue_depth": JOBS.qsize(),
        "in_flight_jobs": sum(1 for j in JOBS.jobs.values() if not j.future.done()),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if not CONFIG["dashboard"].get("enabled", True):
        raise HTTPException(status_code=404)
    refresh = CONFIG["dashboard"].get("refresh_seconds", 5)
    w = WORKER.snapshot()
    pending = sum(1 for j in JOBS.jobs.values() if not j.future.done())
    html = f"""
    <html><head><meta http-equiv="refresh" content="{refresh}">
    <title>GGUF Proxy Dashboard</title>
    <style>
        body {{ font-family: monospace; background:#0d1117; color:#c9d1d9; padding:2rem; }}
        .ok {{ color:#3fb950; }} .bad {{ color:#f85149; }}
        table {{ border-collapse: collapse; }} td, th {{ padding: 4px 12px; text-align:left; }}
    </style></head><body>
    <h2>GGUF Kaggle Proxy Dashboard</h2>
    <table>
      <tr><td>Worker online</td><td class="{'ok' if w['online'] else 'bad'}">{w['online']}</td></tr>
      <tr><td>Worker id</td><td>{w['worker_id']}</td></tr>
      <tr><td>Transport</td><td>{w['transport']}</td></tr>
      <tr><td>Status</td><td>{w['status']}</td></tr>
      <tr><td>Current model</td><td>{w['current_model']}</td></tr>
      <tr><td>Known models</td><td>{', '.join(w['known_models']) or '(none reported yet)'}</td></tr>
      <tr><td>Queue depth</td><td>{JOBS.qsize()}</td></tr>
      <tr><td>Pending jobs (incl. in-flight)</td><td>{pending}</td></tr>
    </table>
    </body></html>
    """
    return HTMLResponse(html)


# --------------------------------------------------------------------------- #
# Worker registration payload (shared by WS hello and long-poll register)
# --------------------------------------------------------------------------- #


def apply_worker_hello(worker_id: str, models: dict, transport: str) -> None:
    WORKER.worker_id = worker_id
    WORKER.models = models or {}
    WORKER.transport = transport
    WORKER.status = "idle"
    WORKER.touch()
    logger.info("Worker '%s' registered via %s. Models: %s",
                worker_id, transport, list(WORKER.models.keys()))


# --------------------------------------------------------------------------- #
# WebSocket transport (primary)
# --------------------------------------------------------------------------- #


@app.websocket("/worker/ws")
async def worker_ws(websocket: WebSocket):
    await websocket.accept()
    authed = False
    worker_id = None
    try:
        # First message must be an auth/hello frame.
        hello_raw = await asyncio.wait_for(websocket.receive_text(), timeout=15)
        hello = json.loads(hello_raw)
        if hello.get("type") != "hello":
            await websocket.close(code=4001)
            return
        worker_id = hello.get("worker_id", "kaggle-worker")
        timestamp = hello.get("timestamp", "")
        signature = hello.get("signature", "")
        if not verify_worker_auth(worker_id, timestamp, signature):
            await websocket.send_text(json.dumps({"type": "auth_failed"}))
            await websocket.close(code=4003)
            return

        if WORKER.is_online() and WORKER.worker_id and WORKER.worker_id != worker_id:
            # Enforce "exactly one active worker" -- reject a second worker
            # while an existing one is alive.
            await websocket.send_text(json.dumps({
                "type": "auth_failed",
                "reason": "another worker is already active",
            }))
            await websocket.close(code=4009)
            return

        authed = True
        WORKER.ws = websocket
        apply_worker_hello(worker_id, hello.get("models", {}), "ws")
        await websocket.send_text(json.dumps({"type": "hello_ack"}))

        heartbeat_timeout = CONFIG["websocket"]["heartbeat_timeout_seconds"]
        ack_timeout = CONFIG["job_delivery_ack_timeout_seconds"]

        async def dispatcher():
            """Pulls jobs off the queue and pushes them to this worker one
            at a time (the GGUF backend processes a single request at a
            time), waiting for the result before moving on."""
            while True:
                job = await JOBS.wait_next_for_delivery(timeout=5)
                if job is None:
                    if not WORKER.is_online():
                        return
                    continue
                WORKER.status = "busy"
                try:
                    await websocket.send_text(json.dumps({
                        "type": "job",
                        "job_id": job.id,
                        "payload": job.payload,
                    }))
                except Exception as e:
                    await JOBS.requeue_or_fail(job, f"send failed: {e}")
                    return
                # Wait (bounded) for this specific job's future to resolve.
                # The receive loop below is what actually completes it.
                deadline = time.time() + job.timeout
                while not job.future.done() and time.time() < deadline:
                    await asyncio.sleep(0.2)
                    if not WORKER.is_online():
                        break
                if not job.future.done():
                    await JOBS.requeue_or_fail(job, "worker did not respond in time")
                WORKER.status = "idle"

        dispatcher_task = asyncio.create_task(dispatcher())

        try:
            while True:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=heartbeat_timeout)
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "heartbeat":
                    WORKER.touch()
                    WORKER.status = msg.get("status", WORKER.status)
                    WORKER.current_model = msg.get("current_model", WORKER.current_model)
                    await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))
                elif mtype == "ack":
                    WORKER.touch()  # job delivery acknowledged, nothing else to do
                elif mtype == "result":
                    WORKER.touch()
                    job_id = msg.get("job_id")
                    JOBS.complete(job_id, msg.get("result"), msg.get("error"))
                elif mtype == "models_update":
                    WORKER.models = msg.get("models", WORKER.models)
                else:
                    logger.debug("Unknown WS message type from worker: %s", mtype)
        finally:
            dispatcher_task.cancel()

    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as e:
        logger.warning("Worker WS session error: %s", e)
    finally:
        if authed and WORKER.worker_id == worker_id:
            WORKER.ws = None
            WORKER.transport = None
            logger.info("Worker '%s' disconnected from WebSocket.", worker_id)


# --------------------------------------------------------------------------- #
# Long-poll transport (fallback)
# --------------------------------------------------------------------------- #


@app.post("/worker/register")
async def worker_register(request: Request):
    body = await request.json()
    worker_id = body.get("worker_id", "kaggle-worker")
    if not verify_worker_auth(worker_id, body.get("timestamp", ""), body.get("signature", "")):
        raise HTTPException(status_code=401, detail="bad worker signature")
    if WORKER.is_online() and WORKER.transport == "ws" and WORKER.worker_id != worker_id:
        raise HTTPException(status_code=409, detail="another worker is already active")
    apply_worker_hello(worker_id, body.get("models", {}), "poll")
    return {"status": "registered"}


@app.get("/worker/poll")
async def worker_poll(worker_id: str, timestamp: str, signature: str):
    if not verify_worker_auth(worker_id, timestamp, signature):
        raise HTTPException(status_code=401, detail="bad worker signature")
    WORKER.touch()
    if WORKER.transport != "ws":
        WORKER.transport = "poll"
    WORKER.status = "idle"
    job = await JOBS.wait_next_for_delivery(timeout=CONFIG["long_poll_timeout_seconds"])
    if job is None:
        return {"job": None}
    WORKER.status = "busy"
    return {"job": {"job_id": job.id, "payload": job.payload}}


@app.post("/worker/result")
async def worker_result(request: Request):
    body = await request.json()
    worker_id = body.get("worker_id", "kaggle-worker")
    if not verify_worker_auth(worker_id, body.get("timestamp", ""), body.get("signature", "")):
        raise HTTPException(status_code=401, detail="bad worker signature")
    WORKER.touch()
    WORKER.status = "idle"
    job_id = body.get("job_id")
    ok = JOBS.complete(job_id, body.get("result"), body.get("error"))
    return {"accepted": ok}


@app.post("/worker/heartbeat")
async def worker_heartbeat(request: Request):
    """Used by the long-poll worker between jobs so it can be marked online
    even while /worker/poll is (deliberately) blocked waiting for work."""
    body = await request.json()
    worker_id = body.get("worker_id", "kaggle-worker")
    if not verify_worker_auth(worker_id, body.get("timestamp", ""), body.get("signature", "")):
        raise HTTPException(status_code=401, detail="bad worker signature")
    WORKER.touch()
    WORKER.status = body.get("status", WORKER.status)
    WORKER.current_model = body.get("current_model", WORKER.current_model)
    if "models" in body:
        WORKER.models = body["models"]
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Background watchdog: fail jobs that were delivered but never completed
# (covers the case of a worker dying mid-job on either transport).
# --------------------------------------------------------------------------- #


async def watchdog_loop():
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for job in list(JOBS.jobs.values()):
            if job.future.done():
                continue
            if job.delivered and job.delivered_at and (now - job.delivered_at) > job.timeout:
                await JOBS.requeue_or_fail(job, "delivery watchdog timeout")
            elif not job.delivered and (now - job.created_at) > job.timeout:
                job.future.set_exception(RuntimeError("timed out waiting in queue"))
        JOBS.cleanup()


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(watchdog_loop())
    logger.info("Proxy server started on %s:%s", CONFIG["host"], CONFIG["port"])


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="info")
