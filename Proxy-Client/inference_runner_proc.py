"""
inference_runner_proc.py
================================================================================
PERSISTENT child-process entrypoint used by inference_client.py to run
llama.cpp generation calls in a KILLABLE, isolated process.

Why this exists
----------------
llama-cpp-python's create_chat_completion()/__call__() is a blocking native
call with no cancellation hook. If a client disconnects or a job times out
while that call is running in-process, there is no way to stop it: it just
keeps burning GPU time and holding the model's single execution context,
blocking every other queued job behind it until it finally returns on its
own.

Running generation in a separate OS process fixes this: the parent
(inference_client.py) can terminate()/kill() this process the instant a
cancellation is needed, which immediately frees the GPU/context. That's the
only reliable way to "force kill" a stuck inference call.

Persistent, not per-job
------------------------
This process loads the model ONCE at startup and then serves jobs one at a
time in a loop for as long as it lives -- it is NOT spawned fresh per job,
because reloading a multi-GB GGUF model for every single request would add
unacceptable latency to normal (non-cancelled) traffic. The parent only
kills and respawns this process when a job actually needs to be cancelled;
every other job reuses the same warm model.

Protocol (line-delimited JSON over stdin/stdout)
-------------------------------------------------
On startup, this process reads ONE "load" line describing how to load the
model:

    {"type": "load", "model_path": "...", "n_ctx": 4096, "n_batch": 512,
     "n_threads": 8, "n_gpu_layers": -1, "tensor_split": [0.5, 0.5] | null,
     "embedding": false}

"embedding" (default false) tells llama.cpp to build the model with
embedding output enabled, which is required before any "embeddings"-kind
job will work -- see the embeddings section below.

and emits either:

    {"type": "loaded"}                      -- ready for jobs
    {"type": "load_error", "error": "..."}  -- fatal, process should be
                                                treated as dead by the parent

After a successful load, it then reads one "job" line at a time in a loop:

    {"type": "job", "job_id": "...", "job": { ...same shape as before... }}

and emits, per job:

    {"type": "token", "job_id": ..., "delta": "...", "finish_reason": null|"stop"}
      (zero or more, streaming chat/completion jobs only)
    {"type": "done", "job_id": ..., "result": {"text": ..., "finish_reason": ...,
                                                 "usage": {...}}}
      (chat/completion jobs)
    {"type": "done", "job_id": ..., "result": {"embeddings": [[...], ...],
                                                 "usage": {...}}}
      (embeddings jobs -- always non-streaming, one vector per input string)
    {"type": "error", "job_id": ..., "error": "..."}

job["kind"] is one of "chat", "completion", or "embeddings". For chat/
completion jobs, job["max_tokens"] may be None/absent, meaning UNLIMITED
generation -- llama-cpp-python itself already treats max_tokens<=0 or None
as "generate until EOS/stop/context limit" (see run_job() below), so this
script passes it straight through rather than substituting its own
default cap.

Embeddings jobs require the model to have been loaded with
{"embedding": true} in the "load" message (see below) -- llama.cpp needs
this flag set at load time to build the model with an embedding-capable
compute graph/pooling head. A model loaded without it will fail any
embeddings job with a clear error rather than silently returning
garbage. Whether a given GGUF actually has good embedding quality still
depends on the model itself (dedicated embedding models like
nomic-embed-text or bge-* work well; a generic chat model's embeddings
are usable but not optimized for retrieval).

then waits for the next "job" line. job_id is included purely so the parent
can sanity-check it's reading events for the job it thinks is running
(this process only ever works on one job at a time).

If the parent kills this process (SIGTERM/SIGKILL) at any point -- during
load, during a job, or while idle between jobs -- the OS tears the whole
thing down immediately, including whatever CUDA context/VRAM it was
holding. There is no cleanup this script needs to do for that case; that's
the entire point of running generation here instead of in-process.
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _read_line() -> dict:
    """Read and parse one JSON line from stdin. Returns {} on EOF (parent
    closed the pipe / this process should exit)."""
    raw = sys.stdin.readline()
    if not raw:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    return json.loads(raw)


def run_embeddings_job(llm, job: dict, job_id: str) -> None:
    """Non-streaming only -- embeddings are a single blocking call, there's
    no token-by-token output to push. Requires the model to have been
    loaded with embedding=True (see main()); llama-cpp-python raises if
    you call create_embedding() on a model that wasn't."""
    inputs = job.get("input") or []
    if isinstance(inputs, str):
        inputs = [inputs]
    try:
        out = llm.create_embedding(input=inputs)
    except Exception as e:
        raise RuntimeError(
            "embeddings_not_supported: this worker's loaded model was not "
            f"initialized with embedding=True, or does not support "
            f"embeddings: {e}") from e
    # llama-cpp-python's create_embedding() returns OpenAI-shaped
    # {"data": [{"embedding": [...], "index": 0}, ...], "usage": {...}}.
    # Sort by index defensively (should already be in order) and hand
    # back just the vectors, in request order, for format_embeddings_
    # response() on the proxy side to wrap into the final OpenAI shape.
    data = sorted(out.get("data", []), key=lambda d: d.get("index", 0))
    vectors = [d["embedding"] for d in data]
    usage = out.get("usage", {})
    _emit({"type": "done", "job_id": job_id,
           "result": {"embeddings": vectors, "usage": usage}})


def run_job(llm, job: dict, job_id: str) -> None:
    kind = job["kind"]

    if kind == "embeddings":
        run_embeddings_job(llm, job, job_id)
        return

    # max_tokens<=0 or None is passed straight through to llama-cpp-python
    # unchanged -- it already treats that as "unlimited, bounded only by
    # n_ctx" (see its own create_chat_completion/__call__ docstrings), so
    # there is no default to substitute here. The old code's
    # `job.get("max_tokens", 512)` used to silently cap every request that
    # didn't explicitly set max_tokens at 512 tokens; that default is gone.
    gen_kwargs = dict(
        temperature=job.get("temperature", 0.8),
        top_p=job.get("top_p", 0.95),
        max_tokens=job.get("max_tokens"),
        stop=job.get("stop"),
        presence_penalty=job.get("presence_penalty", 0.0),
        frequency_penalty=job.get("frequency_penalty", 0.0),
    )
    if job.get("seed") is not None:
        gen_kwargs["seed"] = job["seed"]

    stream = bool(job.get("stream", False))

    if not stream:
        if kind == "chat":
            out = llm.create_chat_completion(messages=job["messages"], **gen_kwargs)
            choice = out["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
        else:
            out = llm(prompt=job["prompt"], **gen_kwargs)
            choice = out["choices"][0]
            text = choice["text"]
            finish_reason = choice.get("finish_reason", "stop")
        usage = out.get("usage", {})
        _emit({"type": "done", "job_id": job_id,
               "result": {"text": text, "finish_reason": finish_reason, "usage": usage}})
        return

    completion_tokens = 0
    if kind == "chat":
        gen = llm.create_chat_completion(messages=job["messages"], stream=True, **gen_kwargs)
        for chunk in gen:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}) or {}
            text = delta.get("content", "") or ""
            finish_reason = choice.get("finish_reason")
            if text:
                completion_tokens += 1
            if text or finish_reason:
                _emit({"type": "token", "job_id": job_id, "delta": text, "finish_reason": finish_reason})
    else:
        gen = llm(prompt=job["prompt"], stream=True, **gen_kwargs)
        for chunk in gen:
            choice = chunk["choices"][0]
            text = choice.get("text", "") or ""
            finish_reason = choice.get("finish_reason")
            if text:
                completion_tokens += 1
            if text or finish_reason:
                _emit({"type": "token", "job_id": job_id, "delta": text, "finish_reason": finish_reason})

    _emit({"type": "done", "job_id": job_id, "result": {"usage": {"completion_tokens": completion_tokens}}})


def main() -> None:
    load_req = _read_line()
    if not load_req or load_req.get("type") != "load":
        _emit({"type": "load_error", "error": "expected a 'load' message as the first line"})
        return

    model_path = load_req["model_path"]
    n_ctx = load_req["n_ctx"]
    n_batch = load_req.get("n_batch", 512)
    n_threads = load_req.get("n_threads") or (os.cpu_count() or 4)
    n_gpu_layers = load_req.get("n_gpu_layers", -1)
    tensor_split = load_req.get("tensor_split")
    # Must be set at load time -- llama.cpp builds a different compute
    # graph/pooling head for embedding output. A model loaded with
    # embedding=False can still run chat/completion jobs fine, but any
    # "embeddings" job against it will fail with a clear error from
    # run_embeddings_job() above rather than silently misbehaving.
    embedding = bool(load_req.get("embedding", False))

    try:
        from llama_cpp import Llama
    except Exception as e:
        _emit({"type": "load_error", "error": f"could not import llama_cpp: {e}"})
        return

    # The n_ctx we're handed here was validated once by the parent's
    # startup probe, but that validation can go stale by the time this
    # (freshly spawned, separate) process actually tries to allocate its
    # own llama_context: VRAM fragments, other Kaggle processes claim
    # memory, or the compute-buffer allocation is simply right on the
    # edge (see "sched_reserve: compute buffer allocation failed,
    # retrying without pipeline parallelism" in worker logs -- that retry
    # already tells you the probe barely fit). Rather than treating a
    # single failed attempt at the parent's cached n_ctx as fatal, retry
    # a few times with a smaller n_ctx, same as the parent's own
    # load_startup_model() OOM-shrink loop.
    min_ctx = load_req.get("min_ctx", 256)
    shrink_factor = load_req.get("oom_shrink_factor", 0.75)
    max_retries = load_req.get("max_load_retries", 3)

    attempts = 0
    last_error = None
    llm = None
    cur_ctx = n_ctx
    while True:
        attempts += 1
        try:
            kwargs = dict(
                model_path=model_path,
                n_ctx=cur_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                embedding=embedding,
                verbose=False,
            )
            if tensor_split:
                llm = Llama(tensor_split=tensor_split, **kwargs)
            else:
                llm = Llama(**kwargs)
            break
        except Exception as e:
            last_error = f"{e}\n{traceback.format_exc()}"
            if attempts >= max_retries + 1 or cur_ctx <= min_ctx:
                _emit({"type": "load_error",
                       "error": f"model load failed in runner after {attempts} attempt(s) "
                                f"(last n_ctx={cur_ctx}): {last_error}"})
                return
            new_ctx = max(min_ctx, int(cur_ctx * shrink_factor))
            sys.stderr.write(
                f"[runner] load failed at n_ctx={cur_ctx} (attempt {attempts}/{max_retries + 1}); "
                f"retrying at n_ctx={new_ctx}\n")
            sys.stderr.flush()
            cur_ctx = new_ctx

    if cur_ctx != n_ctx:
        # Tell the parent what we actually ended up loading with, so its
        # dashboard/state (MODELS.n_ctx) doesn't keep reporting a context
        # size that no longer matches what's actually loaded -- otherwise
        # the next respawn will just repeat this same failed attempt.
        _emit({"type": "loaded", "n_ctx": cur_ctx})
    else:
        _emit({"type": "loaded"})

    # Main job loop: block waiting for the next job, run it, emit results,
    # repeat. Exits cleanly on EOF (parent closed stdin, e.g. during a
    # graceful shutdown) or gets torn down instantly by SIGTERM/SIGKILL at
    # any point (including mid-job) when the parent needs to cancel.
    while True:
        req = _read_line()
        if not req:
            return  # EOF: parent closed the pipe, nothing more to do
        if req.get("type") != "job":
            continue
        job_id = req.get("job_id", "")
        job = req.get("job", {})
        try:
            run_job(llm, job, job_id)
        except Exception as e:
            _emit({"type": "error", "job_id": job_id,
                   "error": f"{e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()