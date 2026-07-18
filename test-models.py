#!/usr/bin/env python3
"""
Controlled OpenAI-compatible load tester for a local proxy.

Features:
- Concurrent chat.completions requests
- Streaming or non-streaming mode
- Real-time dashboard polling and parsing
- Live per-worker routing stats if response headers expose worker id
- Latency, TTFT, throughput, success rate
- CSV/JSON export

Install:
    pip install aiohttp

Examples:
    python loadtest.py --requests 200 --concurrency 16
    python loadtest.py --requests 500 --concurrency 32 --stream
    python loadtest.py --requests 300 --concurrency 24 --weights 30 4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


DEFAULT_PROMPT = "Say hello in one short sentence."
DEFAULT_SYSTEM = "You are a helpful assistant."


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    status: int
    latency_s: float
    ttft_s: Optional[float]
    output_chars: int
    output_words: int
    worker_id: str
    error: str = ""


@dataclass
class DashboardWorker:
    worker_id: str
    online: bool
    transport: str
    status: str
    current_model: str
    known_models: str
    in_flight: int
    queue: int
    gpu: str
    last_hb_s: float


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def safe_p95(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1))
    return vals[idx]


def parse_dashboard_html(html: str) -> Tuple[str, Dict[str, DashboardWorker]]:
    """
    Parses the simple dashboard HTML you pasted.
    Returns (summary_line, workers_by_id)
    """
    summary_match = re.search(
        r'<p class="sub">\s*(.*?)\s*</p>', html, flags=re.IGNORECASE | re.DOTALL
    )
    summary = re.sub(r"\s+", " ", summary_match.group(1)).strip() if summary_match else ""

    # Parse <tr> rows with <td> cells.
    rows = re.findall(r"<tr>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    workers: Dict[str, DashboardWorker] = {}

    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
        cells = [re.sub(r"<.*?>", "", c).strip() for c in cells]
        if len(cells) < 10:
            continue
        # Skip header row
        if cells[0].lower() == "worker id":
            continue

        try:
            worker_id = cells[0]
            online = cells[1].lower() == "true"
            transport = cells[2]
            status = cells[3]
            current_model = cells[4]
            known_models = cells[5]
            in_flight = int(cells[6])
            queue = int(cells[7])
            gpu = cells[8]
            last_hb_s = float(cells[9])
            workers[worker_id] = DashboardWorker(
                worker_id=worker_id,
                online=online,
                transport=transport,
                status=status,
                current_model=current_model,
                known_models=known_models,
                in_flight=in_flight,
                queue=queue,
                gpu=gpu,
                last_hb_s=last_hb_s,
            )
        except Exception:
            continue

    return summary, workers


async def fetch_dashboard(session: aiohttp.ClientSession, url: str) -> Tuple[str, Dict[str, DashboardWorker]]:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        text = await resp.text()
        return parse_dashboard_html(text)


def extract_worker_id(headers: aiohttp.typedefs.LooseHeaders) -> str:
    # Try several likely headers your proxy might expose.
    candidates = [
        "x-worker-id",
        "x-routing-worker",
        "x-proxy-worker",
        "x-upstream-worker",
        "x-served-by",
    ]
    for key in candidates:
        val = headers.get(key) if hasattr(headers, "get") else None
        if val:
            return str(val)
    return "unknown"


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    request_id: int,
    stream: bool,
    timeout_s: float,
) -> RequestResult:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": stream,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }

    started = time.perf_counter()
    ttft_s: Optional[float] = None
    output = []
    worker_id = "unknown"

    try:
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            worker_id = extract_worker_id(resp.headers)

            if not stream:
                data = await resp.json(content_type=None)
                ended = time.perf_counter()
                text = ""
                try:
                    text = data["choices"][0]["message"]["content"] or ""
                except Exception:
                    text = json.dumps(data)[:2000]
                return RequestResult(
                    request_id=request_id,
                    ok=200 <= resp.status < 300,
                    status=resp.status,
                    latency_s=ended - started,
                    ttft_s=None,
                    output_chars=len(text),
                    output_words=len(text.split()),
                    worker_id=worker_id,
                    error="" if 200 <= resp.status < 300 else f"HTTP {resp.status}",
                )

            # Streaming SSE-ish format
            buffer = ""
            first_token_seen = False

            async for chunk_bytes in resp.content.iter_chunked(1024):
                chunk = chunk_bytes.decode("utf-8", errors="ignore")
                buffer += chunk

                # Process line-wise
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                output.append(content)
                                if not first_token_seen:
                                    ttft_s = time.perf_counter() - started
                                    first_token_seen = True
                        except Exception:
                            continue

            ended = time.perf_counter()
            text = "".join(output)
            return RequestResult(
                request_id=request_id,
                ok=200 <= resp.status < 300,
                status=resp.status,
                latency_s=ended - started,
                ttft_s=ttft_s,
                output_chars=len(text),
                output_words=len(text.split()),
                worker_id=worker_id,
                error="" if 200 <= resp.status < 300 else f"HTTP {resp.status}",
            )

    except Exception as e:
        ended = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            ok=False,
            status=0,
            latency_s=ended - started,
            ttft_s=ttft_s,
            output_chars=0,
            output_words=0,
            worker_id=worker_id,
            error=repr(e),
        )


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def bar(value: float, max_value: float, width: int = 24) -> str:
    if max_value <= 0:
        return " " * width
    filled = int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + " " * (width - filled)


def print_live(
    elapsed_s: float,
    results: List[RequestResult],
    dashboard_summary: str,
    workers: Dict[str, DashboardWorker],
    worker_weights: Dict[str, float],
) -> None:
    completed = len(results)
    failed = sum(1 for r in results if not r.ok)
    ok = completed - failed
    latencies = [r.latency_s for r in results if r.ok]
    ttfts = [r.ttft_s for r in results if r.ok and r.ttft_s is not None]
    rps = completed / elapsed_s if elapsed_s > 0 else 0.0

    by_worker = Counter(r.worker_id for r in results if r.ok)
    total_routed = sum(by_worker.values())

    clear_screen()
    print("GGUF Multi-Worker Load Test")
    print("=" * 78)
    print(f"Elapsed:      {elapsed_s:8.1f}s")
    print(f"Completed:    {completed}")
    print(f"Success:      {ok}")
    print(f"Failed:       {failed}")
    print(f"Req/sec:      {rps:8.2f}")
    if latencies:
        print(f"Avg latency:  {safe_mean(latencies):8.3f}s")
        print(f"Median:       {safe_median(latencies):8.3f}s")
        print(f"P95:          {safe_p95(latencies):8.3f}s")
    if ttfts:
        print(f"Avg TTFT:     {safe_mean(ttfts):8.3f}s")
        print(f"P95 TTFT:     {safe_p95(ttfts):8.3f}s")

    print("\nDashboard")
    print("-" * 78)
    print(dashboard_summary or "(no dashboard summary parsed yet)")

    if workers:
        print("\nWorkers")
        print("-" * 78)
        max_q = max((w.queue for w in workers.values()), default=1)
        max_if = max((w.in_flight for w in workers.values()), default=1)
        for wid, w in workers.items():
            routed = by_worker.get(wid, 0)
            actual_share = (routed / total_routed) if total_routed else 0.0
            weight = worker_weights.get(wid, 0.0)
            print(
                f"{wid:>8} | online={str(w.online):5} | {w.status:10} | "
                f"in-flight={w.in_flight:2} | queue={w.queue:2} | hb={w.last_hb_s:4.1f}s | "
                f"gpu={w.gpu}"
            )
            if total_routed:
                ideal_total_weight = sum(worker_weights.values()) or 1.0
                ideal_share = weight / ideal_total_weight if ideal_total_weight > 0 else 0.0
                eff = (actual_share / ideal_share) if ideal_share > 0 else 0.0
                print(
                    f"          routed={routed:4} | actual={actual_share*100:5.1f}% | "
                    f"ideal={ideal_share*100:5.1f}% | balance={eff*100:5.1f}%"
                )
            print(
                f"          queue [{bar(w.queue, max_q)}] {w.queue}   "
                f"in-flight [{bar(w.in_flight, max_if)}] {w.in_flight}"
            )
    else:
        print("\nWorkers")
        print("-" * 78)
        print("(waiting for dashboard parse...)")

    if total_routed:
        print("\nRouting distribution")
        print("-" * 78)
        for wid, count in by_worker.most_common():
            print(f"{wid:>10}: {count:4}  ({count / total_routed * 100:5.1f}%)")

    print("\nTip: Ctrl+C stops the test cleanly.")
    sys.stdout.flush()


async def dashboard_loop(
    session: aiohttp.ClientSession,
    dashboard_url: str,
    state: dict,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            summary, workers = await fetch_dashboard(session, dashboard_url)
            state["dashboard_summary"] = summary
            state["workers"] = workers
        except Exception:
            pass
        await asyncio.sleep(1.0)


async def live_loop(
    state: dict,
    results: List[RequestResult],
    worker_weights: Dict[str, float],
    stop_event: asyncio.Event,
) -> None:
    start = time.perf_counter()
    while not stop_event.is_set():
        elapsed = time.perf_counter() - start
        print_live(
            elapsed_s=elapsed,
            results=results,
            dashboard_summary=state.get("dashboard_summary", ""),
            workers=state.get("workers", {}),
            worker_weights=worker_weights,
        )
        await asyncio.sleep(1.0)


async def run_load_test(args: argparse.Namespace) -> List[RequestResult]:
    timeout_s = args.timeout
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=60)
    results: List[RequestResult] = []
    state: dict = {"dashboard_summary": "", "workers": {}}
    stop_event = asyncio.Event()

    worker_weights: Dict[str, float] = {}
    if len(args.weights) >= 2:
        worker_weights = {f"worker{i+1}": float(w) for i, w in enumerate(args.weights)}
    else:
        worker_weights = {"worker1": 30.0, "worker2": 4.0}

    async with aiohttp.ClientSession(connector=connector) as session:
        dash_task = asyncio.create_task(dashboard_loop(session, args.dashboard_url, state, stop_event))
        live_task = asyncio.create_task(live_loop(state, results, worker_weights, stop_event))

        sem = asyncio.Semaphore(args.concurrency)

        async def bounded_request(i: int) -> None:
            async with sem:
                r = await one_request(
                    session=session,
                    url=args.url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    user_prompt=args.user_prompt,
                    request_id=i,
                    stream=args.stream,
                    timeout_s=timeout_s,
                )
                results.append(r)

        try:
            tasks = [asyncio.create_task(bounded_request(i)) for i in range(args.requests)]
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        finally:
            stop_event.set()
            await asyncio.gather(dash_task, live_task, return_exceptions=True)

    return results


def export_results(results: List[RequestResult], json_path: str, csv_path: str) -> None:
    as_dicts = [dataclasses.asdict(r) for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(as_dicts, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "ok",
                "status",
                "latency_s",
                "ttft_s",
                "output_chars",
                "output_words",
                "worker_id",
                "error",
            ],
        )
        writer.writeheader()
        for row in as_dicts:
            writer.writerow(row)


def final_summary(results: List[RequestResult]) -> None:
    ok_results = [r for r in results if r.ok]
    failed_results = [r for r in results if not r.ok]
    latencies = [r.latency_s for r in ok_results]
    ttfts = [r.ttft_s for r in ok_results if r.ttft_s is not None]
    by_worker = Counter(r.worker_id for r in ok_results)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"Total requests: {len(results)}")
    print(f"Successful:     {len(ok_results)}")
    print(f"Failed:         {len(failed_results)}")
    if latencies:
        print(f"Avg latency:    {safe_mean(latencies):.3f}s")
        print(f"Median:         {safe_median(latencies):.3f}s")
        print(f"P95 latency:    {safe_p95(latencies):.3f}s")
    if ttfts:
        print(f"Avg TTFT:       {safe_mean(ttfts):.3f}s")
        print(f"P95 TTFT:       {safe_p95(ttfts):.3f}s")

    if by_worker:
        total = sum(by_worker.values())
        print("\nWorker routing share:")
        for wid, count in by_worker.most_common():
            print(f"  {wid:>10}: {count:5} ({count / total * 100:5.1f}%)")

    if failed_results:
        print("\nFailures:")
        for r in failed_results[:10]:
            print(f"  #{r.request_id}: status={r.status} error={r.error}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--url",
        default="https://kaggle-inference-proxy.onrender.com/v1/chat/completions",
        help="Chat completions endpoint",
    )
    p.add_argument(
        "--dashboard-url",
        default="https://kaggle-inference-proxy.onrender.com/dashboard",
        help="Proxy dashboard URL",
    )
    p.add_argument("--api-key", default="sk-change-me-client-key", help="Bearer token")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        help="Model id",
    )
    p.add_argument("--requests", type=int, default=100, help="Total number of requests")
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent requests")
    p.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses for TTFT measurement",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout seconds",
    )
    p.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM,
        help="System prompt",
    )
    p.add_argument(
        "--user-prompt",
        default=DEFAULT_PROMPT,
        help="User prompt",
    )
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=[30.0, 4.0],
        help="Worker capacity weights, e.g. 30 4",
    )
    p.add_argument(
        "--json-out",
        default="loadtest_results.json",
        help="Output JSON file",
    )
    p.add_argument(
        "--csv-out",
        default="loadtest_results.csv",
        help="Output CSV file",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        results = asyncio.run(run_load_test(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    export_results(results, args.json_out, args.csv_out)
    final_summary(results)

    print(f"\nSaved: {args.json_out}")
    print(f"Saved: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())