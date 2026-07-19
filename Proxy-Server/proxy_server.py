"""
proxy_server_multi.py
================================================================================
Public-facing, OpenAI-compatible API proxy — MULTI-WORKER ROUTER.

This is the multi-inference-server evolution of proxy_server.py. Instead of
tracking a single active worker, this process can accept connections from
many Kaggle-hosted inference workers simultaneously (one model loaded per
worker), and routes each incoming client request to whichever connected
worker is:

    1. Actually serving the requested model
    2. Least busy right now (fewest in-flight jobs, lowest GPU utilization)
    3. Most recently heard from (tie-break), then round-robin as a final
       deterministic tie-break

Clients only ever see a normal, stateless OpenAI-compatible API:

    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions
    POST /v1/embeddings          (not supported -> clean OpenAI-style error)
    GET  /health
    GET  /dashboard

Workers connect OUTBOUND to this server (this server never dials into
Kaggle) via one of two transports:

    * WebSocket  (ws://.../worker/ws)   <- preferred, low latency, push-based
    * Long-poll  (POST /worker/register, GET /worker/poll,
                  POST /worker/result, POST /worker/heartbeat,
                  POST /worker/deregister)  <- fallback transport

Unlike the single-worker reference, MANY workers may be connected and
"active" at the same time. Each worker has its own job queue; a job is
routed to exactly one worker's queue based on the scoring function below and
never migrates between workers mid-flight (especially not mid-stream).

CANCELLATION
------------
A job's computed result is only useful if someone is still waiting for it.
The proxy now actively cancels jobs on the worker side (not just locally)
whenever that stops being true:

  * The client disconnects while a request is in flight (checked for both
    streaming and non-streaming requests).
  * A job's request_timeout_seconds elapses without a result (existing
    watchdog behavior), which previously only resolved the job locally with
    an error while the worker kept computing in the background.
  * A worker connection drops mid-job (existing requeue_or_fail behavior)
    and the job is ultimately failed rather than retried.

In each case the proxy now sends `{"type": "cancel", "job_id": ...}` to the
owning worker over its WebSocket (if connected), so the worker can hard-kill
whatever is computing that job instead of letting it run to completion and
block every job queued up behind it. For long-poll workers (no push
channel), a pending cancellation is instead piggybacked onto the next
heartbeat response as `{"cancel_job_id": "..."}`, which the worker polls at
its regular heartbeat interval -- this bounds cancellation latency on that
transport to one heartbeat interval, which is the best a pure
request/response fallback transport can offer; WS remains the transport to
prefer for prompt cancellation.

Config loading:
    Config is normally read from (and, if missing, written to) the JSON file
    at PROXY_CONFIG_PATH (default "proxy_server_config.json"). If the
    PROXY_SERVER_CONFIG environment variable is set, it takes priority over
    the file entirely: its value is expected to be a base64-encoded JSON
    document containing the config (or a partial config to overlay onto the
    defaults), and no file is read or written in that case.

Worker reconnect handling:
    If a worker's connection drops while it still has queued or in-flight
    jobs, the proxy gives it a grace window (worker_reconnect_grace_seconds,
    default 30s) to reconnect (same worker_id, either transport). If the
    worker has not come back online by the time that window elapses, ALL of
    its remaining jobs (queued and in-flight) are cancelled with an error,
    and the worker is removed from the registry entirely.

Run:
    pip install fastapi "uvicorn[standard]" websockets
    python proxy_server_multi.py
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import itertools
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.environ.get("PROXY_CONFIG_PATH", "proxy_server_config.json")
CONFIG_ENV_VAR = "PROXY_SERVER_CONFIG"

DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "client_api_keys": ["sk-your-api-key-here"],
    "worker_shared_secret": "change-me-worker-secret",
    "max_queue_size_per_worker": 100,
    "request_timeout_seconds": 300,
    "long_poll_timeout_seconds": 40,
    "job_delivery_ack_timeout_seconds": 15,
    "worker_offline_after_seconds": 90,
    "worker_reconnect_grace_seconds": 30,
    "auth_timestamp_skew_seconds": 300,
    "client_disconnect_check_interval_seconds": 1.0,
    "websocket": {
        "heartbeat_interval_seconds": 20,
        "heartbeat_timeout_seconds": 60,
    },
    "retry": {
        "max_job_retries": 1,
    },
    "routing": {
        "prefer_least_busy_gpu": True,
        "fallback_to_round_robin": True,
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


def _merge_config(defaults: dict, overlay: dict) -> dict:
    """Deep-merge `overlay` onto `defaults` (one level of nested dicts, which
    is all this config shape uses), without mutating either input."""
    merged = json.loads(json.dumps(defaults))
    merged.update(overlay)
    for k, v in defaults.items():
        if isinstance(v, dict) and isinstance(overlay.get(k), dict):
            merged[k] = {**v, **overlay[k]}
    return merged


def load_config_from_env(var_name: str, defaults: dict) -> Optional[dict]:
    """If `var_name` is set, decode it as base64-encoded JSON and merge it
    onto `defaults`. Returns None if the env var isn't set, so the caller
    can fall back to file-based config. Raises a clear error if the env var
    is set but malformed, rather than silently ignoring it."""
    raw = os.environ.get(var_name)
    if raw is None or raw.strip() == "":
        return None
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as e:
        raise RuntimeError(
            f"Environment variable {var_name} is set but is not valid "
            f"base64: {e}") from e
    try:
        parsed = json.loads(decoded.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError(
            f"Environment variable {var_name} decoded from base64 but is "
            f"not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Environment variable {var_name} must decode to a JSON object, "
            f"got {type(parsed).__name__}.")
    print(f"[proxy_server_multi] Loaded config from ${var_name} "
          f"(base64-encoded JSON env var); ignoring '{CONFIG_PATH}'.")
    return _merge_config(defaults, parsed)


def load_or_create_config(path: str, defaults: dict) -> dict:
    """Load config with the following precedence:

    1. PROXY_SERVER_CONFIG env var (base64-encoded JSON) if set -- no file
       is read or written in this case.
    2. JSON config file at `path`, backfilling any missing keys from
       `defaults` without clobbering user edits.
    3. If neither is present, write out a default config file to `path` and
       use the defaults, same as before.
    """
    from_env = load_config_from_env(CONFIG_ENV_VAR, defaults)
    if from_env is not None:
        return from_env

    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(defaults, f, indent=2)
        print(f"[proxy_server_multi] No config found at '{path}' and "
              f"${CONFIG_ENV_VAR} is not set. A default config has been "
              f"created. Please review/edit it (API keys, secrets, ports) "
              f"before exposing this server publicly.")
        return json.loads(json.dumps(defaults))
    with open(path, "r") as f:
        cfg = json.load(f)
    return _merge_config(defaults, cfg)


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
logger = logging.getLogger("proxy_server_multi")

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
# Job dataclass (unchanged shape from the single-worker reference)
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    kind: str                      # "chat", "completions"
    payload: dict                  # normalized job payload sent to worker
    worker_id: Optional[str] = None            # which worker this job was routed to
    created_at: float = field(default_factory=time.time)
    timeout: float = 300.0
    future: "asyncio.Future" = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    delivered: bool = False
    delivered_at: Optional[float] = None
    last_activity_at: Optional[float] = None   # refreshed on every streamed token
    retries: int = 0
    stream: bool = False
    stream_queue: Optional["asyncio.Queue"] = None   # only set for true (WS) streaming jobs
    cancel_requested: bool = False             # True once we've asked the worker to abort this job

    # -- Return-path liveness -------------------------------------------- #
    # abandoned=True means the client that wanted this job's output is
    # confirmed gone (its request.is_disconnected() came back true during
    # our once-a-second liveness check -- see watch_job_liveness() below).
    # Once set, this job must never be handed to a worker: if it's still
    # sitting in worker.queue, the dispatcher discards it on pop instead of
    # delivering it; if it's already delivered/running, the owning worker
    # is told to hard-kill it. This is the single source of truth the
    # dispatcher consults -- there is no other path by which a job with no
    # live return path is allowed to occupy the runner.
    abandoned: bool = False
    liveness_task: "Optional[asyncio.Task]" = None


# --------------------------------------------------------------------------- #
# Per-worker state
# --------------------------------------------------------------------------- #


class WorkerState:
    """Tracks one connected inference worker (whichever transport it is
    using) along with its own private job queue."""

    def __init__(self, worker_id: str) -> None:
        self.worker_id: str = worker_id
        self.transport: Optional[str] = None        # "ws" or "poll"
        self.ws: Optional[WebSocket] = None
        self.last_heartbeat: float = 0.0
        self.models: Dict[str, Any] = {}
        self.status: str = "unknown"                 # idle / busy / loading_model / unknown
        self.current_model: Optional[str] = None
        self.in_flight_jobs: int = 0
        self.gpu_stats: dict = {}                     # {"total_mb", "used_mb", "free_mb", "utilization_percent"}
        self.n_ctx: Optional[int] = None               # loaded context size, reported via heartbeat
        self.lock = asyncio.Lock()

        # Per-worker job queue + job map (jobs routed to this worker only).
        self.queue: "asyncio.Queue[str]" = asyncio.Queue(
            maxsize=CONFIG.get("max_queue_size_per_worker", 100))
        self.jobs: Dict[str, Job] = {}

        # Bookkeeping for the WS dispatcher task tied to this connection.
        self.dispatcher_task: Optional["asyncio.Task"] = None

        # Job id the worker should be told to cancel on its next heartbeat
        # response, for workers on the long-poll transport (which has no
        # push channel to deliver a cancel immediately). WS workers instead
        # get an immediate out-of-band "cancel" message -- see
        # request_cancel() below -- and don't use this field.
        self.pending_cancel_job_id: Optional[str] = None

        # Set by /kill-all (see request_kill_all() below) for long-poll
        # workers, which have no push channel: relayed on the worker's next
        # heartbeat/poll response, same pattern as pending_cancel_job_id.
        # WS workers instead get an immediate out-of-band "kill_all"
        # message and don't use this field.
        self.pending_kill_all: bool = False

        # Set the instant we positively learn the worker's connection is
        # gone (WS close, explicit deregister, or heartbeat staleness during
        # a sweep). Used by the reconnect-grace watchdog below to decide
        # when a disconnected worker has been gone too long and should be
        # torn down entirely, rather than left around indefinitely with a
        # dead queue. Cleared (set back to None) the moment the worker
        # successfully re-registers, on either transport.
        self.disconnected_at: Optional[float] = None

    def is_online(self) -> bool:
        if self.last_heartbeat == 0:
            return False
        return (time.time() - self.last_heartbeat) < CONFIG["worker_offline_after_seconds"]

    def touch(self) -> None:
        self.last_heartbeat = time.time()
        self.disconnected_at = None

    def mark_offline(self) -> None:
        """Immediately mark the worker offline instead of waiting for the
        worker_offline_after_seconds grace window to elapse. Used whenever we
        positively know the connection is gone (WS close, explicit
        deregister, or heartbeat staleness during a sweep). Also stamps
        disconnected_at (if not already stamped), which starts the
        reconnect-grace clock consulted by reap_unreconnected_workers()."""
        self.last_heartbeat = 0.0
        self.status = "unknown"
        if self.disconnected_at is None:
            self.disconnected_at = time.time()

    async def request_cancel(self, job_id: str, reason: str) -> None:
        """Ask this worker to hard-kill whatever is running for job_id, right
        now if possible. On WS this is an immediate out-of-band push; on
        long-poll it's queued to ride along with the next heartbeat
        response (see /worker/heartbeat), which bounds latency to one
        heartbeat interval on that transport."""
        job = self.jobs.get(job_id)
        if job is not None:
            job.cancel_requested = True
        if self.transport == "ws" and self.ws is not None:
            try:
                # Shares self.lock with the main recv loop's heartbeat_ack
                # sends and the dispatcher task's job sends (see worker_ws)
                # -- all three run concurrently on the same websocket, and
                # unserialized concurrent send_text() calls can interleave
                # partial frames on the wire, corrupting the connection.
                async with self.lock:
                    await self.ws.send_text(json.dumps({"type": "cancel", "job_id": job_id}))
                logger.info("Sent cancel for job %s to worker '%s' (%s) over WS.",
                            job_id, self.worker_id, reason)
            except Exception as e:
                logger.warning("Failed to send cancel for job %s to worker '%s': %s",
                                job_id, self.worker_id, e)
        else:
            # Long-poll: no push channel. Stash it; /worker/heartbeat will
            # hand it back on the worker's next heartbeat call. Only keep
            # the most recent request -- a worker running one job at a time
            # only ever needs to cancel the one it's currently working on.
            self.pending_cancel_job_id = job_id
            logger.info("Queued cancel for job %s for worker '%s' (%s); will be delivered "
                        "on next long-poll heartbeat.", job_id, self.worker_id, reason)

    async def request_kill_all(self, reason: str) -> None:
        """Tell this worker to unconditionally hard-kill and reset its
        local runner subprocess, regardless of which job_id (if any) it
        currently thinks it's running. Unlike request_cancel(), this isn't
        keyed to a specific job -- it's an emergency reset for when the
        normal per-job cancel path has stopped working (e.g. a wedged
        subprocess that never released its job lock, so the worker keeps
        reporting busy and further cancels for it get silently ignored
        because the job_id no longer matches whatever the worker thinks is
        current)."""
        if self.transport == "ws" and self.ws is not None:
            try:
                async with self.lock:
                    await self.ws.send_text(json.dumps({"type": "kill_all"}))
                logger.info("Sent kill_all to worker '%s' (%s) over WS.", self.worker_id, reason)
            except Exception as e:
                logger.warning("Failed to send kill_all to worker '%s': %s", self.worker_id, e)
        else:
            self.pending_kill_all = True
            logger.info("Queued kill_all for worker '%s' (%s); will be delivered on next "
                        "long-poll heartbeat.", self.worker_id, reason)

    def qsize(self) -> int:
        return self.queue.qsize()

    def snapshot(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "online": self.is_online(),
            "transport": self.transport,
            "status": self.status,
            "current_model": self.current_model,
            "n_ctx": self.n_ctx,
            "in_flight_jobs": self.in_flight_jobs,
            "gpu_stats": self.gpu_stats,
            "known_models": list(self.models.keys()),
            "queue_depth": self.qsize(),
            "last_heartbeat_age_seconds": round(time.time() - self.last_heartbeat, 1)
            if self.last_heartbeat else None,
            "disconnected_for_seconds": round(time.time() - self.disconnected_at, 1)
            if self.disconnected_at else None,
        }


# --------------------------------------------------------------------------- #
# Worker registry + routing
# --------------------------------------------------------------------------- #


class WorkerRegistry:
    """Owns the Dict[worker_id -> WorkerState] and all job bookkeeping across
    every worker. Routing decisions (which worker gets a given request) live
    here too, since they need a global view of all workers."""

    def __init__(self) -> None:
        self.workers: Dict[str, WorkerState] = {}
        self._rr_counter = itertools.count()  # round-robin tiebreaker source
        self._registry_lock = asyncio.Lock()

    async def get_or_create(self, worker_id: str) -> WorkerState:
        async with self._registry_lock:
            w = self.workers.get(worker_id)
            if w is None:
                w = WorkerState(worker_id)
                self.workers[worker_id] = w
            return w

    def get(self, worker_id: str) -> Optional[WorkerState]:
        return self.workers.get(worker_id)

    def all_workers(self) -> List[WorkerState]:
        return list(self.workers.values())

    def online_workers(self) -> List[WorkerState]:
        return [w for w in self.workers.values() if w.is_online()]

    def all_known_models(self) -> Dict[str, dict]:
        """Union of models across all currently-online workers, for
        GET /v1/models. If multiple workers serve the same model id, the
        last one wins for metadata purposes (metadata itself is usually
        trivial/empty)."""
        merged: Dict[str, dict] = {}
        for w in self.online_workers():
            for model_id, meta in w.models.items():
                merged[model_id] = meta
        return merged

    def total_queue_depth(self) -> int:
        return sum(w.qsize() for w in self.workers.values())

    def total_in_flight(self) -> int:
        return sum(
            1
            for w in self.workers.values()
            for j in w.jobs.values()
            if not j.future.done()
        )

    def score_worker(self, worker: WorkerState) -> Tuple[float, float, float, int]:
        """Lower score = better. Used for routing decisions.

        priority = (in_flight_jobs, gpu_utilization_percent, heartbeat_age,
                    round_robin_counter)

        The round-robin counter is only consulted as the final tiebreaker
        when everything else is equal (e.g. two freshly-registered, idle
        workers serving the same model)."""
        utilization = worker.gpu_stats.get("utilization_percent", 0) or 0
        in_flight = worker.in_flight_jobs
        recency = time.time() - worker.last_heartbeat if worker.last_heartbeat else float("inf")
        rr = next(self._rr_counter) if CONFIG["routing"].get("fallback_to_round_robin", True) else 0
        return (in_flight, utilization, recency, rr)

    def route_request(self, model: str) -> Optional[WorkerState]:
        """Pick the best online worker currently serving `model`."""
        candidates = [
            w for w in self.workers.values()
            if w.is_online() and model in w.models
        ]
        if not candidates:
            return None
        scored = [(self.score_worker(w), w) for w in candidates]
        scored.sort(key=lambda pair: pair[0])
        return scored[0][1]

    async def submit_job(self, worker: WorkerState, kind: str, payload: dict,
                          stream: bool, request: Optional[Request] = None) -> Job:
        if worker.queue.full():
            raise HTTPException(status_code=429, detail=openai_error(
                "Selected inference server's queue is full. Please retry shortly.",
                "rate_limit_error"))
        job = Job(
            id=str(uuid.uuid4()),
            kind=kind,
            payload=payload,
            worker_id=worker.worker_id,
            timeout=CONFIG["request_timeout_seconds"],
            stream=stream,
        )
        if stream:
            job.stream_queue = asyncio.Queue()
        worker.jobs[job.id] = job
        worker.in_flight_jobs += 1
        await worker.queue.put(job.id)
        logger.info("Routed job %s (kind=%s, model=%s) -> worker '%s' "
                    "(in_flight=%d, gpu_util=%s%%)",
                    job.id, kind, payload.get("model"), worker.worker_id,
                    worker.in_flight_jobs, worker.gpu_stats.get("utilization_percent"))
        # Start the once-a-second return-path liveness check for the FULL
        # lifetime of this job (queued AND delivered), not just while it's
        # actively running. This is the single rule that replaces the old
        # per-endpoint disconnect watchers: from the moment a job exists,
        # if there is ever no live client connection to deliver its output
        # to, the job is marked abandoned and torn down immediately --
        # cancelled on the worker if it's already delivered/running, and
        # made ineligible for delivery if it's still sitting in the queue.
        if request is not None:
            job.liveness_task = asyncio.create_task(
                self.watch_job_liveness(worker, job, request))
        return job

    async def watch_job_liveness(self, worker: WorkerState, job: Job,
                                  request: Request) -> None:
        """Polls request.is_disconnected() once a second for as long as
        `job` is neither finished nor already abandoned. This is the ONE
        rule the proxy enforces for every job it holds, queued or
        delivered: no live return path -> halt immediately. The moment
        disconnection is observed, the job is marked abandoned (so the
        dispatcher will never deliver it if it hasn't been already) and,
        if it has already been delivered to a worker, that worker is told
        to hard-kill it right away."""
        interval = CONFIG.get("client_disconnect_check_interval_seconds", 1.0)
        try:
            while not job.future.done() and not job.abandoned:
                await asyncio.sleep(interval)
                if job.future.done():
                    return
                if await request.is_disconnected():
                    job.abandoned = True
                    logger.info(
                        "No live return path for job %s (client disconnected); "
                        "marking abandoned and halting on worker '%s'.",
                        job.id, worker.worker_id)
                    await self.cancel_job_on_worker(
                        worker, job, "no return path: client disconnected")
                    if job.stream_queue is not None and not job.future.done():
                        await job.stream_queue.put({"error": "client disconnected"})
                        await job.stream_queue.put(None)
                    if not job.future.done():
                        job.future.set_exception(
                            RuntimeError("client disconnected: no return path for job output"))
                        worker.in_flight_jobs = max(0, worker.in_flight_jobs - 1)
                    return
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning("Liveness watcher for job %s crashed: %s", job.id, e)

    def find_job(self, job_id: str) -> Tuple[Optional[WorkerState], Optional[Job]]:
        """Locate a job (and its owning worker) by id, searching across all
        workers. Used by long-poll result/heartbeat endpoints which only
        carry worker_id + job_id, not a direct WorkerState reference."""
        for w in self.workers.values():
            j = w.jobs.get(job_id)
            if j is not None:
                return w, j
        return None, None

    def pop_next_for_delivery(self, worker: WorkerState) -> Optional[Job]:
        """Non-blocking pop from this worker's queue. Skips (drains without
        delivering) any job that has already been marked abandoned by its
        liveness watcher -- a job with no live return path must never reach
        the worker, queued or not."""
        while True:
            try:
                job_id = worker.queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
            job = worker.jobs.get(job_id)
            if job is None or job.future.done():
                continue
            if job.abandoned:
                logger.info("Dropping queued job %s before delivery: no live return path.", job_id)
                continue
            job.delivered = True
            job.delivered_at = time.time()
            job.last_activity_at = job.delivered_at
            return job

    async def wait_next_for_delivery(self, worker: WorkerState, timeout: float) -> Optional[Job]:
        """Blocking (up to timeout) pop from this worker's queue, used by
        both the WS dispatcher loop and the long-poll endpoint. Skips any
        job already marked abandoned by its liveness watcher, same as
        pop_next_for_delivery -- a job that has no live return path never
        gets handed to the worker."""
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                job_id = await asyncio.wait_for(worker.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            job = worker.jobs.get(job_id)
            if job is None or job.future.done():
                continue
            if job.abandoned:
                logger.info("Dropping queued job %s before delivery: no live return path.", job_id)
                continue
            job.delivered = True
            job.delivered_at = time.time()
            job.last_activity_at = job.delivered_at
            return job

    def mark_activity(self, worker: WorkerState, job_id: str) -> None:
        """Called whenever a streamed token arrives, so long-running streams
        aren't mistaken for a stalled/dead worker by the watchdog."""
        job = worker.jobs.get(job_id)
        if job is not None:
            job.last_activity_at = time.time()

    def complete(self, worker: WorkerState, job_id: str, result: Optional[dict],
                 error: Optional[str]) -> bool:
        job = worker.jobs.get(job_id)
        if job is None or job.future.done():
            return False  # unknown or duplicate delivery -- ignore safely
        if error:
            job.future.set_exception(RuntimeError(error))
        else:
            job.future.set_result(result)
        worker.in_flight_jobs = max(0, worker.in_flight_jobs - 1)
        if worker.pending_cancel_job_id == job_id:
            worker.pending_cancel_job_id = None
        return True

    async def cancel_job_on_worker(self, worker: WorkerState, job: Job, reason: str) -> None:
        """Tell the owning worker to hard-kill whatever compute is currently
        happening for `job`, because nobody will ever consume its result
        (client disconnected, or it's already been failed/timed-out
        locally). Safe to call even if the job was never delivered yet (the
        worker will simply report "not running" and ignore it) or has
        already completed (worker-side cancel is a no-op post-completion)."""
        if job.cancel_requested:
            return  # already asked; avoid spamming duplicate cancel messages
        await worker.request_cancel(job.id, reason)

    async def requeue_or_fail(self, worker: WorkerState, job: Job, reason: str) -> None:
        """Called by the watchdog (or by an immediate disconnect handler)
        when a delivered job never got a result. Jobs are only ever
        requeued onto the SAME worker's queue (never migrated to a
        different worker), matching the "streaming job MUST stay with same
        worker" requirement -- for non-streaming jobs this still means a
        retry only makes sense if that worker is expected to come back
        (e.g. a transient send failure), otherwise it will simply fail once
        retries are exhausted or the worker is confirmed offline.

        Whenever a job is being permanently failed here (not requeued),
        the owning worker is also told to cancel/hard-kill it, since the
        result -- if the worker is even still computing it -- will never be
        read by anyone. Requeued jobs are NOT cancelled: they haven't been
        given up on, they're being retried on the same worker."""
        if job.future.done():
            return
        # A streaming job that has already sent tokens to the client cannot
        # be safely retried (the client already received a partial
        # response) -- fail it outright and unblock the SSE consumer.
        if job.stream_queue is not None:
            logger.error("Failing streaming job %s on worker '%s': %s",
                         job.id, worker.worker_id, reason)
            await job.stream_queue.put({"error": reason})
            await job.stream_queue.put(None)
            job.future.set_exception(RuntimeError(reason))
            worker.in_flight_jobs = max(0, worker.in_flight_jobs - 1)
            await self.cancel_job_on_worker(worker, job, reason)
            return
        if job.retries < CONFIG["retry"]["max_job_retries"] and worker.is_online():
            job.retries += 1
            job.delivered = False
            job.delivered_at = None
            job.last_activity_at = None
            logger.warning("Requeuing job %s on worker '%s' after failure: %s",
                           job.id, worker.worker_id, reason)
            await worker.queue.put(job.id)
        else:
            logger.error("Failing job %s on worker '%s' permanently: %s",
                         job.id, worker.worker_id, reason)
            job.future.set_exception(RuntimeError(reason))
            worker.in_flight_jobs = max(0, worker.in_flight_jobs - 1)
            await self.cancel_job_on_worker(worker, job, reason)

    async def fail_in_flight_jobs_for_worker(self, worker: WorkerState, reason: str) -> None:
        """Immediately fail/requeue every job that was already delivered to
        this (now-gone) worker connection, instead of waiting for the
        watchdog to notice via its timeout. Jobs still only queued (never
        delivered) are left in this worker's queue -- if the worker comes
        back online within the reconnect grace window they'll be picked up
        then; if it doesn't come back within that window,
        cancel_all_jobs_for_worker() below will cancel those too and the
        worker will be removed entirely."""
        in_flight = [j for j in worker.jobs.values() if not j.future.done() and j.delivered]
        for job in in_flight:
            await self.requeue_or_fail(worker, job, reason)

    async def cancel_all_jobs_for_worker(self, worker: WorkerState, reason: str) -> int:
        """Forcefully cancels every still-pending job owned by this worker
        (both delivered/in-flight and merely queued-but-undelivered), used
        when a disconnected worker has exceeded its reconnect grace window
        and is about to be removed from the registry entirely. Unlike
        requeue_or_fail, this never retries -- the worker is being deleted,
        so there is nowhere left to requeue to. Returns the number of jobs
        cancelled."""
        cancelled = 0
        for job in list(worker.jobs.values()):
            if job.future.done():
                continue
            if job.stream_queue is not None:
                await job.stream_queue.put({"error": reason})
                await job.stream_queue.put(None)
            job.future.set_exception(RuntimeError(reason))
            cancelled += 1
        worker.in_flight_jobs = 0
        # Drain the queue itself so nothing lingers referencing job ids that
        # are about to be discarded along with the worker.
        while not worker.queue.empty():
            try:
                worker.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        return cancelled

    async def kill_all(self, reason: str = "kill-all requested") -> dict:
        """Emergency stop: empties every worker's queue, fails every
        in-flight job across the whole fleet, and tells every worker
        (regardless of transport) to unconditionally hard-kill and reset
        its local runner subprocess -- not just whatever job_id it thinks
        is current, which is what makes this a useful escape hatch when
        the normal per-job cancel path has gotten stuck (see
        request_kill_all's docstring). Intentionally does not distinguish
        online/offline workers: an offline worker still gets its jobs
        cancelled locally here, and will pick up the kill_all the moment
        it reconnects and calls /worker/heartbeat or /worker/poll (both
        check pending_kill_all)."""
        total_cancelled = 0
        workers_signalled: List[str] = []
        for worker in self.all_workers():
            total_cancelled += await self.cancel_all_jobs_for_worker(worker, reason)
            await worker.request_kill_all(reason)
            workers_signalled.append(worker.worker_id)
        return {
            "status": "ok",
            "jobs_cancelled": total_cancelled,
            "workers_signalled": workers_signalled,
        }

    async def reap_unreconnected_workers(self, grace_seconds: float, reason: str) -> List[str]:
        """Watchdog helper: for every worker that has been continuously
        disconnected for longer than `grace_seconds`, cancel all of its
        pending/in-flight jobs and remove it from the registry entirely
        (rather than leaving a dead entry around whose queue nothing will
        ever service again). Returns the list of worker_ids that were
        removed."""
        now = time.time()
        to_remove = [
            w for w in self.workers.values()
            if w.disconnected_at is not None
            and (now - w.disconnected_at) >= grace_seconds
        ]
        removed_ids: List[str] = []
        for worker in to_remove:
            # Re-check online status defensively -- a worker could have
            # reconnected (which clears disconnected_at via touch()) between
            # building the list above and getting here.
            if worker.is_online() or worker.disconnected_at is None:
                continue
            n = await self.cancel_all_jobs_for_worker(worker, reason)
            if n:
                logger.warning(
                    "Worker '%s' did not reconnect within %.0fs of "
                    "disconnecting; cancelled %d pending job(s) and "
                    "removed the worker from the registry.",
                    worker.worker_id, grace_seconds, n)
            else:
                logger.info(
                    "Worker '%s' did not reconnect within %.0fs of "
                    "disconnecting; removed the worker from the registry.",
                    worker.worker_id, grace_seconds)
            self.workers.pop(worker.worker_id, None)
            removed_ids.append(worker.worker_id)
        return removed_ids

    def cleanup(self, max_age: float = 3600) -> None:
        now = time.time()
        for w in self.workers.values():
            stale = [jid for jid, j in w.jobs.items()
                     if j.future.done() and (now - j.created_at) > max_age]
            for jid in stale:
                del w.jobs[jid]

    def prune_dead_workers(self, max_age: float = 24 * 3600) -> None:
        """Drop bookkeeping for workers that have been offline for a very
        long time, so the registry doesn't grow unbounded across many
        Kaggle session churns. Only prunes workers with no in-flight/queued
        jobs left. In normal operation, reap_unreconnected_workers() (30s
        grace window, or as configured) will have already removed and
        cancelled any disconnected worker with pending jobs long before
        this much slower long-horizon prune ever runs; this remains as a
        backstop for workers that disconnect cleanly with no pending work."""
        now = time.time()
        dead = []
        for wid, w in self.workers.items():
            if w.is_online():
                continue
            if w.in_flight_jobs > 0 or w.qsize() > 0:
                continue
            if w.last_heartbeat and (now - w.last_heartbeat) > max_age:
                dead.append(wid)
        for wid in dead:
            del self.workers[wid]
            logger.info("Pruned long-dead worker '%s' from registry.", wid)


REGISTRY = WorkerRegistry()


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(title="GGUF-Kaggle OpenAI-Compatible Multi-Worker Proxy")

if CONFIG["cors"].get("enabled"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CONFIG["cors"].get("allow_origins", ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --------------------------------------------------------------------------- #
# OpenAI response formatting helpers (unchanged from reference)
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
    """Used only when the winning worker is on the long-poll fallback
    transport, which cannot push individual tokens as they're generated. We
    compute the full result and then emit it as a single SSE 'delta' chunk
    followed by [DONE]. This keeps stream=True clients working (just
    without real incremental output) whenever that worker isn't on WS."""
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


async def real_stream_generator(job: Job, model: str, kind: str, request: Request,
                                 worker: WorkerState):
    """True token-by-token SSE stream for jobs delivered over a worker's
    WebSocket. Reads from job.stream_queue as that worker's WS handler
    feeds it (see 'stream_chunk' / 'stream_done' / 'error' below), and
    yields OpenAI-compatible chunk frames as soon as each token arrives.
    The job (and therefore the stream) never migrates to a different
    worker mid-flight.

    Disconnect handling is NOT done here anymore -- job.liveness_task
    (started in submit_job) already polls request.is_disconnected() once a
    second for this job's entire lifetime and will mark job.abandoned +
    cancel it on the worker the moment the client is gone, which unblocks
    the stream_queue.get() below via the future/queue sentinel this
    generator is already watching. This loop only needs to notice that
    happened and stop yielding.
    """
    role_sent = False
    try:
        while True:
            try:
                item = await asyncio.wait_for(job.stream_queue.get(), timeout=job.timeout)
            except asyncio.TimeoutError:
                err = openai_error(
                    "Timed out waiting for streamed tokens from the worker.",
                    "server_error")
                yield f"data: {json.dumps(err)}\n\n"
                await REGISTRY.cancel_job_on_worker(worker, job, "stream token timeout")
                break

            if item is None:
                break  # normal end-of-stream sentinel (includes the
                       # abandoned-job sentinel pushed by watch_job_liveness)

            if "error" in item:
                err = openai_error(f"Worker streaming error: {item['error']}", "server_error")
                yield f"data: {json.dumps(err)}\n\n"
                break

            delta_text = item.get("delta", "") or ""
            finish_reason = item.get("finish_reason")

            if kind == "chat":
                if not role_sent:
                    first = {
                        "id": f"chatcmpl-{job.id}", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(first)}\n\n"
                    role_sent = True
                delta_obj = {"content": delta_text} if delta_text else {}
                chunk = {
                    "id": f"chatcmpl-{job.id}", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": delta_obj, "finish_reason": finish_reason}],
                }
            else:
                chunk = {
                    "id": f"cmpl-{job.id}", "object": "text_completion",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "text": delta_text, "logprobs": None,
                                 "finish_reason": finish_reason}],
                }
            yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        if job.liveness_task is not None and not job.liveness_task.done():
            job.liveness_task.cancel()
        yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Shared request handling / routing
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


def route_or_503(model: str) -> WorkerState:
    """Find the best worker for `model`, or raise a clean 503."""
    worker = REGISTRY.route_request(model)
    if worker is None:
        # Distinguish "model unknown anywhere" from "model exists but all
        # workers serving it are offline" for a clearer error message.
        known_anywhere = any(model in w.models for w in REGISTRY.all_workers())
        if known_anywhere:
            msg = (f"No inference server currently has model '{model}' "
                   f"online. It is configured on a worker that is not "
                   f"currently connected.")
        else:
            msg = f"No inference server currently has model '{model}' loaded."
        raise HTTPException(status_code=503, detail=openai_error(msg, "server_error"))
    return worker


async def run_job_and_wait(worker: WorkerState, kind: str, payload: dict,
                            request: Request) -> Tuple[Job, dict]:
    """Non-streaming (or poll-fallback) path: submit a job to a specific
    worker's queue and block until that worker returns the full result.

    Disconnect handling is NOT done here anymore -- job.liveness_task
    (started inside submit_job, since it's passed `request`) already polls
    request.is_disconnected() once a second for this job's ENTIRE
    lifetime, including however long it sits queued before a worker ever
    picks it up. The moment it sees the client is gone, it marks the job
    abandoned, cancels it on the worker if already delivered, and resolves
    job.future with an exception -- which is exactly what unblocks the
    await below. This function just needs to translate that outcome into
    an HTTP response.
    """
    job = await REGISTRY.submit_job(worker, kind, payload, stream=False, request=request)
    try:
        result = await asyncio.wait_for(job.future, timeout=job.timeout)
    except asyncio.TimeoutError:
        await REGISTRY.cancel_job_on_worker(worker, job, "request_timeout_seconds elapsed")
        raise HTTPException(status_code=504, detail=openai_error(
            "Timed out waiting for the inference worker to respond.",
            "server_error"))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=openai_error(
            f"Worker failed to process the request: {e}", "server_error"))
    finally:
        if job.liveness_task is not None and not job.liveness_task.done():
            job.liveness_task.cancel()
    return job, result


def worker_supports_real_streaming(worker: WorkerState) -> bool:
    """True token streaming requires the persistent WebSocket transport --
    long-polling has no way to push individual tokens as they're produced."""
    return worker.is_online() and worker.transport == "ws"


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

    worker = route_or_503(model)

    if stream and worker_supports_real_streaming(worker):
        payload = {
            "kind": "chat", "model": model, "messages": messages, "stream": True,
            "params": normalize_generation_params(body),
        }
        job = await REGISTRY.submit_job(worker, "chat", payload, stream=True, request=request)
        return StreamingResponse(
            real_stream_generator(job, model, "chat", request, worker),
            media_type="text/event-stream",
        )

    # Non-streaming, or the chosen worker is currently on the long-poll
    # fallback transport (no true incremental push available there).
    payload = {
        "kind": "chat", "model": model, "messages": messages, "stream": False,
        "params": normalize_generation_params(body),
    }
    job, result = await run_job_and_wait(worker, "chat", payload, request)
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

    worker = route_or_503(model)

    if stream and worker_supports_real_streaming(worker):
        payload = {
            "kind": "completion", "model": model, "prompt": prompt, "stream": True,
            "params": normalize_generation_params(body),
        }
        job = await REGISTRY.submit_job(worker, "completions", payload, stream=True, request=request)
        return StreamingResponse(
            real_stream_generator(job, model, "completion", request, worker),
            media_type="text/event-stream",
        )

    payload = {
        "kind": "completion", "model": model, "prompt": prompt, "stream": False,
        "params": normalize_generation_params(body),
    }
    job, result = await run_job_and_wait(worker, "completions", payload, request)
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
            "Only chat/completions models are available via connected workers.",
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
        for alias in REGISTRY.all_known_models().keys()
    ]
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    REGISTRY.cleanup()
    workers_snapshot = [w.snapshot() for w in REGISTRY.all_workers()]
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workers": workers_snapshot,
        "queue_depth": REGISTRY.total_queue_depth(),
        "in_flight_jobs": REGISTRY.total_in_flight(),
    }


@app.get("/kill-all")
async def kill_all():
    """Emergency stop button: empties every worker's queue, fails every
    in-flight job across the whole fleet, and tells every worker
    (WS: immediately; long-poll: on its next heartbeat/poll) to
    unconditionally hard-kill and reset its local runner subprocess --
    regardless of which job_id it currently thinks it owns. Meant as a
    manual escape hatch for exactly the failure mode where a wedged
    subprocess never released its local job lock, so per-job cancels for
    later jobs keep getting silently ignored (job_id no longer matches
    whatever the worker thinks is "current") and the worker just sits
    reporting busy forever.

    Deliberately unauthenticated (no client API key, no worker HMAC
    signature) -- it's an operator-only reset switch on the same trust
    boundary as the machine hosting this proxy, not a client-facing API
    surface, and requiring auth here would defeat the point of a
    break-glass endpoint if that auth is ever what's stuck/misconfigured.
    If this proxy is ever exposed somewhere untrusted, consider putting
    it behind a network-level restriction (e.g. firewall/reverse-proxy
    rule) rather than relying on this route to check credentials."""
    result = await REGISTRY.kill_all()
    logger.warning("kill-all invoked: %s", result)
    return result


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if not CONFIG["dashboard"].get("enabled", True):
        raise HTTPException(status_code=404)
    refresh = CONFIG["dashboard"].get("refresh_seconds", 5)
    workers = REGISTRY.all_workers()
    workers_sorted = sorted(workers, key=lambda w: w.worker_id)

    rows = []
    for w in workers_sorted:
        snap = w.snapshot()
        status_class = "ok" if snap["online"] else "bad"
        gpu = snap["gpu_stats"] or {}
        gpu_str = (f"{gpu.get('used_mb', '?')}/{gpu.get('total_mb', '?')} MB "
                   f"({gpu.get('utilization_percent', '?')}%)") if gpu else "—"
        n_ctx_str = f"{snap['n_ctx']:,}" if snap.get("n_ctx") else "—"
        rows.append(f"""
      <tr>
        <td>{snap['worker_id']}</td>
        <td class="{status_class}">{snap['online']}</td>
        <td>{snap['transport']}</td>
        <td>{snap['status']}</td>
        <td>{snap['current_model'] or '—'}</td>
        <td>{n_ctx_str}</td>
        <td>{', '.join(snap['known_models']) or '(none reported yet)'}</td>
        <td>{snap['in_flight_jobs']}</td>
        <td>{snap['queue_depth']}</td>
        <td>{gpu_str}</td>
        <td>{snap['last_heartbeat_age_seconds']}</td>
      </tr>""")

    total_queue = REGISTRY.total_queue_depth()
    total_in_flight = REGISTRY.total_in_flight()
    online_count = sum(1 for w in workers if w.is_online())

    html = f"""
    <html><head><meta http-equiv="refresh" content="{refresh}">
    <title>Multi-Worker Proxy Dashboard</title>
    <style>
        body {{ font-family: monospace; background:#0d1117; color:#c9d1d9; padding:2rem; }}
        .ok {{ color:#3fb950; }} .bad {{ color:#f85149; }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ padding: 6px 12px; text-align:left; border-bottom: 1px solid #21262d; }}
        th {{ color: #8b949e; text-transform: uppercase; font-size: 0.8em; }}
        .summary td {{ padding: 4px 12px; }}
        h2 {{ margin-bottom: 0.2em; }}
        .sub {{ color: #8b949e; margin-top:0; }}
    </style></head><body>
    <h2>GGUF Kaggle Multi-Worker Proxy Dashboard</h2>
    <p class="sub">{online_count}/{len(workers)} workers online &middot;
       total queue depth {total_queue} &middot; total in-flight {total_in_flight}</p>
    <table>
      <tr>
        <th>Worker ID</th><th>Online</th><th>Transport</th><th>Status</th>
        <th>Current Model</th><th>Loaded n_ctx</th><th>Known Models</th><th>In-Flight</th>
        <th>Queue</th><th>GPU</th><th>Last HB (s)</th>
      </tr>
      {''.join(rows) if rows else '<tr><td colspan="11">No workers have registered yet.</td></tr>'}
    </table>
    </body></html>
    """
    return HTMLResponse(html)


# --------------------------------------------------------------------------- #
# Worker registration payload (shared by WS hello and long-poll register)
# --------------------------------------------------------------------------- #


def apply_worker_hello(worker: WorkerState, models: dict, transport: str) -> None:
    worker.models = models or {}
    worker.transport = transport
    worker.status = "idle"
    worker.touch()
    logger.info("Worker '%s' registered via %s. Models: %s",
                worker.worker_id, transport, list(worker.models.keys()))


# --------------------------------------------------------------------------- #
# WebSocket transport (primary)
# --------------------------------------------------------------------------- #


@app.websocket("/worker/ws")
async def worker_ws(websocket: WebSocket):
    await websocket.accept()
    authed = False
    worker_id = None
    worker: Optional[WorkerState] = None
    try:
        # First message must be an auth/hello frame.
        hello_raw = await asyncio.wait_for(websocket.receive_text(), timeout=15)
        hello = json.loads(hello_raw)
        if hello.get("type") != "hello":
            await websocket.close(code=4001)
            return
        worker_id = hello.get("worker_id")
        if not worker_id:
            await websocket.send_text(json.dumps({
                "type": "auth_failed", "reason": "worker_id is required",
            }))
            await websocket.close(code=4001)
            return
        timestamp = hello.get("timestamp", "")
        signature = hello.get("signature", "")
        if not verify_worker_auth(worker_id, timestamp, signature):
            await websocket.send_text(json.dumps({"type": "auth_failed"}))
            await websocket.close(code=4003)
            return

        worker = await REGISTRY.get_or_create(worker_id)

        if worker.is_online() and worker.ws is not None:
            # This specific worker_id already has an active connection --
            # reject the new one rather than silently stealing the slot.
            # (Unlike the single-worker reference, DIFFERENT worker_ids are
            # always allowed to be active simultaneously; this check only
            # guards against the same worker_id double-connecting.)
            await websocket.send_text(json.dumps({
                "type": "auth_failed",
                "reason": f"worker_id '{worker_id}' already has an active connection",
            }))
            await websocket.close(code=4009)
            return

        authed = True
        worker.ws = websocket
        apply_worker_hello(worker, hello.get("models", {}), "ws")
        await websocket.send_text(json.dumps({"type": "hello_ack"}))

        heartbeat_timeout = CONFIG["websocket"]["heartbeat_timeout_seconds"]

        # worker.lock guards every websocket.send_text() for the rest of
        # this connection's life. dispatcher() below runs as its own
        # asyncio task and pushes "job" frames whenever work is queued,
        # the main recv loop (further down) concurrently sends
        # "heartbeat_ack" frames, and request_cancel() (WorkerState, above)
        # can push an out-of-band "cancel" frame at any moment from a
        # third task entirely -- all three write to the SAME websocket.
        # Unserialized concurrent send_text() calls can interleave partial
        # frames on the wire, corrupting the connection. That corruption
        # (not real network jitter) is what was showing up as spurious
        # "worker disconnected mid-request" / keepalive-timeout failures on
        # the worker side during active streaming. Every send from here on
        # must go through send() below instead of calling
        # websocket.send_text() directly.
        async def send(obj: dict) -> None:
            async with worker.lock:
                await websocket.send_text(json.dumps(obj))

        async def dispatcher(w: WorkerState):
            """Pulls jobs off THIS worker's queue and pushes them to it one
            at a time (a single GGUF backend processes one request at a
            time), waiting for the result before moving on. For streaming
            jobs, 'done' means the worker sent stream_done/error -- the
            individual tokens themselves are delivered separately via the
            job's stream_queue and consumed directly by the SSE response."""
            while True:
                job = await REGISTRY.wait_next_for_delivery(w, timeout=5)
                if job is None:
                    if not w.is_online() or w.ws is not websocket:
                        return
                    continue
                w.status = "busy"
                try:
                    await send({
                        "type": "job",
                        "job_id": job.id,
                        "payload": job.payload,
                    })
                except Exception as e:
                    await REGISTRY.requeue_or_fail(w, job, f"send failed: {e}")
                    return
                # Wait for the job's future to resolve. The window is
                # rolling (based on last_activity_at) rather than a single
                # fixed deadline, so a long but actively-streaming
                # generation doesn't get killed just for taking a while.
                # Note: job.liveness_task resolves job.future immediately
                # (not after job.timeout) the moment it detects no live
                # return path, so an abandoned job falls out of this loop
                # on the very next 0.2s tick rather than waiting out the
                # full timeout window.
                while not job.future.done():
                    await asyncio.sleep(0.2)
                    if not w.is_online() or w.ws is not websocket:
                        break
                    last_activity = job.last_activity_at or job.delivered_at or time.time()
                    if time.time() - last_activity > job.timeout:
                        break
                if not job.future.done():
                    await REGISTRY.requeue_or_fail(w, job, "worker did not respond in time")
                w.status = "idle"

        dispatcher_task = asyncio.create_task(dispatcher(worker))
        worker.dispatcher_task = dispatcher_task

        try:
            while True:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=heartbeat_timeout)
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "heartbeat":
                    worker.touch()
                    worker.status = msg.get("status", worker.status)
                    worker.current_model = msg.get("current_model", worker.current_model)
                    if "n_ctx" in msg:
                        worker.n_ctx = msg.get("n_ctx")
                    gpu_stats = msg.get("gpu_stats")
                    if isinstance(gpu_stats, dict):
                        worker.gpu_stats = gpu_stats
                    await send({"type": "heartbeat_ack"})
                elif mtype == "ack":
                    worker.touch()  # job delivery acknowledged, nothing else to do
                elif mtype == "result":
                    worker.touch()
                    job_id = msg.get("job_id")
                    REGISTRY.complete(worker, job_id, msg.get("result"), msg.get("error"))
                elif mtype == "stream_chunk":
                    # A single token/delta for an in-progress streaming job.
                    worker.touch()
                    job_id = msg.get("job_id")
                    REGISTRY.mark_activity(worker, job_id)
                    job = worker.jobs.get(job_id)
                    if job is not None and job.stream_queue is not None:
                        await job.stream_queue.put({
                            "delta": msg.get("delta", ""),
                            "finish_reason": msg.get("finish_reason"),
                        })
                elif mtype == "stream_done":
                    # Streaming job finished successfully -- unblock the SSE
                    # consumer with the end-of-stream sentinel and resolve
                    # the job future (carries aggregated usage, if any).
                    worker.touch()
                    job_id = msg.get("job_id")
                    job = worker.jobs.get(job_id)
                    if job is not None and job.stream_queue is not None:
                        await job.stream_queue.put(None)
                    REGISTRY.complete(worker, job_id, msg.get("result", {}), None)
                elif mtype == "error":
                    # Out-of-band error for a job that may be mid-stream.
                    worker.touch()
                    job_id = msg.get("job_id")
                    job = worker.jobs.get(job_id)
                    if job is not None and job.stream_queue is not None:
                        await job.stream_queue.put({"error": msg.get("error", "unknown error")})
                        await job.stream_queue.put(None)
                    REGISTRY.complete(worker, job_id, None, msg.get("error", "unknown error"))
                elif mtype == "models_update":
                    worker.models = msg.get("models", worker.models)
                else:
                    logger.debug("Unknown WS message type from worker '%s': %s",
                                worker_id, mtype)
        finally:
            dispatcher_task.cancel()

    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as e:
        logger.warning("Worker '%s' WS session error: %s", worker_id, e)
    finally:
        if authed and worker is not None and worker.ws is websocket:
            worker.ws = None
            worker.transport = None
            # Mark offline THE INSTANT the socket is gone -- don't wait for
            # worker_offline_after_seconds. The dispatcher task tied to this
            # connection has already been cancelled above, so nothing will
            # ever pull queued jobs or resolve in-flight ones for this
            # connection; without this, is_online() would keep reporting
            # True for up to worker_offline_after_seconds, so new requests
            # would be routed to it and then hang, and any job that was
            # already delivered would just sit until its own (much longer)
            # per-job timeout expired. mark_offline() also stamps
            # disconnected_at, which starts the reconnect-grace clock used
            # by the watchdog to decide when to cancel remaining jobs and
            # remove this worker entirely if it never comes back.
            worker.mark_offline()
            logger.info("Worker '%s' disconnected from WebSocket.", worker_id)
            await REGISTRY.fail_in_flight_jobs_for_worker(
                worker, "worker disconnected mid-request")


# --------------------------------------------------------------------------- #
# Long-poll transport (fallback)
# --------------------------------------------------------------------------- #


@app.post("/worker/register")
async def worker_register(request: Request):
    body = await request.json()
    worker_id = body.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id is required")
    if not verify_worker_auth(worker_id, body.get("timestamp", ""), body.get("signature", "")):
        raise HTTPException(status_code=401, detail="bad worker signature")

    worker = await REGISTRY.get_or_create(worker_id)
    if worker.is_online() and worker.transport == "ws":
        # This worker_id already has a live WebSocket connection; don't let
        # a stray long-poll registration for the same id silently steal or
        # confuse routing. Other worker_ids are unaffected.
        raise HTTPException(status_code=409, detail=(
            f"worker_id '{worker_id}' already has an active WebSocket connection"))
    apply_worker_hello(worker, body.get("models", {}), "poll")
    return {"status": "registered"}


@app.get("/worker/poll")
async def worker_poll(worker_id: str, timestamp: str, signature: str):
    if not verify_worker_auth(worker_id, timestamp, signature):
        raise HTTPException(status_code=401, detail="bad worker signature")
    worker = REGISTRY.get(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=(
            "worker_id not registered -- call /worker/register first"))
    worker.touch()
    if worker.transport != "ws":
        worker.transport = "poll"
    worker.status = "idle"
    if worker.pending_kill_all:
        worker.pending_kill_all = False
        return {"job": None, "kill_all": True}
    job = await REGISTRY.wait_next_for_delivery(
        worker, timeout=CONFIG["long_poll_timeout_seconds"])
    if job is None:
        return {"job": None}
    worker.status = "busy"
    return {"job": {"job_id": job.id, "payload": job.payload}}


@app.post("/worker/result")
async def worker_result(request: Request):
    body = await request.json()
    worker_id = body.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id is required")
    if not verify_worker_auth(worker_id, body.get("timestamp", ""), body.get("signature", "")):
        raise HTTPException(status_code=401, detail="bad worker signature")
    worker = REGISTRY.get(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="unknown worker_id")
    worker.touch()
    worker.status = "idle"
    job_id = body.get("job_id")
    ok = REGISTRY.complete(worker, job_id, body.get("result"), body.get("error"))
    return {"accepted": ok}


@app.post("/worker/heartbeat")
async def worker_heartbeat(request: Request):
    """Used by long-poll workers between jobs so they're marked online even
    while /worker/poll is (deliberately) blocked waiting for work. Also
    carries GPU stats and loaded n_ctx, since long-poll workers have no
    other channel to push them outside of a job cycle.

    The response may include `cancel_job_id`: if the proxy has a pending
    cancellation queued for this worker (see WorkerState.request_cancel),
    it's handed back here so the worker can hard-kill that job on its next
    check -- this is the long-poll equivalent of the immediate WS "cancel"
    push, just bounded to one heartbeat interval of latency instead of
    being instant.
    """
    body = await request.json()
    worker_id = body.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id is required")
    if not verify_worker_auth(worker_id, body.get("timestamp", ""), body.get("signature", "")):
        raise HTTPException(status_code=401, detail="bad worker signature")
    worker = REGISTRY.get(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=(
            "worker_id not registered -- call /worker/register first"))
    worker.touch()
    worker.status = body.get("status", worker.status)
    worker.current_model = body.get("current_model", worker.current_model)
    if "n_ctx" in body:
        worker.n_ctx = body.get("n_ctx")
    gpu_stats = body.get("gpu_stats")
    if isinstance(gpu_stats, dict):
        worker.gpu_stats = gpu_stats
    if "models" in body:
        worker.models = body["models"]

    response: Dict[str, Any] = {"status": "ok"}
    if worker.pending_cancel_job_id:
        response["cancel_job_id"] = worker.pending_cancel_job_id
        worker.pending_cancel_job_id = None
    if worker.pending_kill_all:
        response["kill_all"] = True
        worker.pending_kill_all = False
    return response


@app.post("/worker/deregister")
async def worker_deregister(request: Request):
    """Explicit 'I'm shutting down' signal. Both transports can call this on
    a clean shutdown (e.g. Ctrl+C in a Kaggle notebook) so the proxy learns
    about the disconnect immediately instead of relying purely on the
    WebSocket close event or the long-poll heartbeat timeout."""
    body = await request.json()
    worker_id = body.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id is required")
    if not verify_worker_auth(worker_id, body.get("timestamp", ""), body.get("signature", "")):
        raise HTTPException(status_code=401, detail="bad worker signature")
    worker = REGISTRY.get(worker_id)
    if worker is not None:
        worker.ws = None
        worker.transport = None
        worker.mark_offline()
        logger.info("Worker '%s' explicitly deregistered.", worker_id)
        await REGISTRY.fail_in_flight_jobs_for_worker(worker, "worker deregistered")
    return {"status": "deregistered"}


# --------------------------------------------------------------------------- #
# Background watchdog: fail jobs that were delivered but never completed
# (covers the case of a worker dying mid-job on either transport, as a
# backstop for anything the immediate-disconnect handlers above don't catch
# -- e.g. a long-poll worker that vanishes without ever calling
# /worker/deregister). Now iterates every worker's job map, not a single
# global one. It also enforces the reconnect grace window: any worker that
# has been continuously disconnected for longer than
# worker_reconnect_grace_seconds (default 30s) has ALL of its remaining
# jobs cancelled and is removed from the registry outright, rather than
# lingering with a queue nothing will ever service.
# --------------------------------------------------------------------------- #


async def watchdog_loop():
    reconnect_grace = CONFIG.get("worker_reconnect_grace_seconds", 30)
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for worker in REGISTRY.all_workers():
            for job in list(worker.jobs.values()):
                if job.future.done():
                    continue
                if job.delivered:
                    last_activity = job.last_activity_at or job.delivered_at or now
                    if (now - last_activity) > job.timeout:
                        await REGISTRY.requeue_or_fail(worker, job, "delivery watchdog timeout")
                elif (now - job.created_at) > job.timeout:
                    if job.stream_queue is not None:
                        await job.stream_queue.put({"error": "timed out waiting in queue"})
                        await job.stream_queue.put(None)
                    job.future.set_exception(RuntimeError("timed out waiting in queue"))
                    worker.in_flight_jobs = max(0, worker.in_flight_jobs - 1)
                    await REGISTRY.cancel_job_on_worker(worker, job, "timed out waiting in queue")
            # A worker whose heartbeat has gone stale (e.g. a long-poll
            # worker that stopped calling /worker/heartbeat or /worker/poll
            # without ever hitting /worker/deregister) should be swept to
            # offline here too, so routing stops sending it new work even
            # before worker_offline_after_seconds would naturally expire on
            # its own via is_online()'s own check. (is_online() already
            # handles the expiry math; this block only handles failing its
            # in-flight jobs once that expiry has occurred, and stamping
            # disconnected_at so the reconnect-grace reaper below can
            # eventually pick it up too.)
            if not worker.is_online() and worker.in_flight_jobs > 0:
                await REGISTRY.fail_in_flight_jobs_for_worker(
                    worker, "worker heartbeat expired")
                if worker.disconnected_at is None:
                    worker.disconnected_at = now
        # Any worker that has now been disconnected continuously for at
        # least `reconnect_grace` seconds gets its remaining jobs cancelled
        # (queued ones included, not just in-flight) and is dropped from
        # the registry entirely -- it does not get to sit around waiting
        # for a reconnect that isn't guaranteed to come.
        await REGISTRY.reap_unreconnected_workers(
            reconnect_grace, "worker did not reconnect within the grace period")
        REGISTRY.cleanup()
        REGISTRY.prune_dead_workers()


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(watchdog_loop())
    logger.info("Multi-worker proxy server started on %s:%s",
                CONFIG["host"], CONFIG["port"])


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="info")