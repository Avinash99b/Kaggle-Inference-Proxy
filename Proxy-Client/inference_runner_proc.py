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
     "n_threads": 8, "n_gpu_layers": -1, "tensor_split": [0.5, 0.5] | null}

and emits either:

    {"type": "loaded"}                      -- ready for jobs
    {"type": "load_error", "error": "..."}  -- fatal, process should be
                                                treated as dead by the parent

After a successful load, it then reads one "job" line at a time in a loop:

    {"type": "job", "job_id": "...", "job": { ...same shape as before... }}

and emits, per job:

    {"type": "token", "job_id": ..., "delta": "...", "finish_reason": null|"stop"}
      (zero or more, streaming jobs only)
    {"type": "done", "job_id": ..., "result": {"text": ..., "finish_reason": ...,
                                                 "usage": {...}}}
    {"type": "error", "job_id": ..., "error": "..."}

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


def run_job(llm, job: dict, job_id: str) -> None:
    gen_kwargs = dict(
        temperature=job.get("temperature", 0.8),
        top_p=job.get("top_p", 0.95),
        max_tokens=job.get("max_tokens", 512),
        stop=job.get("stop"),
        presence_penalty=job.get("presence_penalty", 0.0),
        frequency_penalty=job.get("frequency_penalty", 0.0),
    )
    if job.get("seed") is not None:
        gen_kwargs["seed"] = job["seed"]

    kind = job["kind"]
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

    try:
        from llama_cpp import Llama
    except Exception as e:
        _emit({"type": "load_error", "error": f"could not import llama_cpp: {e}"})
        return

    try:
        kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        if tensor_split:
            llm = Llama(tensor_split=tensor_split, **kwargs)
        else:
            llm = Llama(**kwargs)
    except Exception as e:
        _emit({"type": "load_error", "error": f"model load failed in runner: {e}\n{traceback.format_exc()}"})
        return

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