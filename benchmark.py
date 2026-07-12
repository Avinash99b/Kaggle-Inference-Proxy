#!/usr/bin/env python3
"""
Benchmark an OpenAI-compatible /v1/chat/completions endpoint.

Fixes the earlier issue where tokens/sec stayed blank by:
- requesting stream_options={"include_usage": true} for streamed requests
- extracting usage from the final streamed chunk when the server supports it
- falling back to tokenizer-based counting when usage is missing

Features:
- multiple prompt suites
- streaming TTFT and chunk-gap measurement
- per-test and overall summaries
- exact usage when available
- approximate token counting fallback when needed
- optional CSV export
- optional non-streaming mode for exact usage-only benchmarking

Example:
  python benchmark.py \
    --api-key sk-... \
    --model coding-qwen3-coder-30b-a3b

If you have a HF tokenizer for your backend, pass:
  --hf-tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class TestCase:
    name: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.95
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class RunResult:
    test_name: str
    run_index: int
    status_code: int
    headers_time_s: float
    ttft_s: Optional[float]
    total_time_s: float
    generation_time_s: Optional[float]
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]
    tok_per_sec: Optional[float]
    ms_per_token: Optional[float]
    avg_chunk_gap_s: Optional[float]
    p95_chunk_gap_s: Optional[float]
    finish_reason: Optional[str]
    error: Optional[str]
    output_text: str
    token_count_source: str


def build_test_suite(long_context_repeat: int = 1200) -> List[TestCase]:
    long_blob = ("The quick brown fox jumps over the lazy dog. " * long_context_repeat).strip()
    return [
        TestCase(
            name="Greeting",
            temperature=0.7,
            max_tokens=96,
            messages=[{"role": "user", "content": "Hello! Introduce yourself in one paragraph."}],
        ),
        TestCase(
            name="Coding",
            temperature=0.2,
            max_tokens=512,
            messages=[{"role": "user", "content": (
                "Write a Python implementation of an LRU Cache supporting O(1) get() and put(). "
                "Include comments and a small example."
            )}],
        ),
        TestCase(
            name="Refactoring",
            temperature=0.2,
            max_tokens=512,
            messages=[{"role": "user", "content": (
                "Refactor this code for readability and performance:\n\n"
                "def fib(n):\n"
                "    if n<=1:\n"
                "        return n\n"
                "    return fib(n-1)+fib(n-2)\n"
            )}],
        ),
        TestCase(
            name="Math",
            temperature=0.1,
            max_tokens=512,
            messages=[{"role": "user", "content": (
                "Find the smallest positive integer divisible by every number from 1 to 20. "
                "Explain your reasoning."
            )}],
        ),
        TestCase(
            name="JSON",
            temperature=0.0,
            max_tokens=512,
            messages=[{"role": "user", "content": (
                "Generate a valid JSON object representing a bookstore with 20 books. "
                "Only output valid JSON."
            )}],
        ),
        TestCase(
            name="Summarization",
            temperature=0.3,
            max_tokens=256,
            messages=[{"role": "user", "content": (
                "Summarize the causes and consequences of World War I in under 250 words."
            )}],
        ),
        TestCase(
            name="Extraction",
            temperature=0.0,
            max_tokens=256,
            messages=[{"role": "user", "content": (
                "Extract every email address, phone number and URL from the following text:\n\n"
                "John Doe\n"
                "Email: john@example.com\n"
                "Phone: +1-555-123-4567\n"
                "Website: https://example.com\n\n"
                "Jane\n"
                "Email: jane@company.org\n"
            )}],
        ),
        TestCase(
            name="Creative Writing",
            temperature=0.9,
            max_tokens=700,
            messages=[{"role": "user", "content": (
                "Write a 500-word sci-fi story involving time travel and AI."
            )}],
        ),
        TestCase(
            name="Long Context",
            temperature=0.2,
            max_tokens=256,
            messages=[{"role": "user", "content": (
                f"{long_blob}\n\nNow summarize the text in a few sentences."
            )}],
        ),
    ]


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def mean_or_none(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def safe_float(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def safe_int(value: Optional[int]) -> str:
    return "-" if value is None else str(value)


class TokenCounter:
    """
    Best-effort token counter:
    1) HF tokenizer if provided
    2) tiktoken if installed
    3) whitespace fallback
    """
    def __init__(self, hf_tokenizer_name: Optional[str], tiktoken_model: str):
        self.source = "none"
        self.tokenizer = None
        self._mode = "approx"

        if hf_tokenizer_name:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name, trust_remote_code=True)
                self.source = f"hf:{hf_tokenizer_name}"
                self._mode = "hf"
                return
            except Exception:
                pass

        try:
            import tiktoken  # type: ignore
            try:
                self.tokenizer = tiktoken.encoding_for_model(tiktoken_model)
                self.source = f"tiktoken:{tiktoken_model}"
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.source = "tiktoken:cl100k_base"
            self._mode = "tiktoken"
            return
        except Exception:
            self.tokenizer = None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._mode == "hf" and self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        if self._mode == "tiktoken" and self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return max(1, len(text.split()))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        if self._mode == "hf" and self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "apply_chat_template"):
                    tokens = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors=None,
                    )
                    if isinstance(tokens, list):
                        return len(tokens)
            except Exception:
                pass

        rendered = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            rendered.append(f"{role}: {content}")
        rendered.append("ASSISTANT:")
        return self.count_text("\n".join(rendered))


def parse_openai_sse_lines(response) -> Tuple[str, Dict[str, Any], float, Optional[float], Optional[float], Optional[str]]:
    output_parts: List[str] = []
    usage: Dict[str, Any] = {}
    finish_reason: Optional[str] = None
    first_token_time: Optional[float] = None
    last_chunk_time: Optional[float] = None
    chunk_gaps: List[float] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break

        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue

        if "choices" in obj and obj["choices"]:
            choice = obj["choices"][0]
            delta = choice.get("delta", {}) or {}
            text = delta.get("content", "") or ""
            if text:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                if last_chunk_time is not None:
                    chunk_gaps.append(now - last_chunk_time)
                last_chunk_time = now
                output_parts.append(text)

            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr

        if "usage" in obj and isinstance(obj["usage"], dict):
            usage = obj["usage"]

    output_text = "".join(output_parts)
    avg_gap = mean_or_none(chunk_gaps)
    p95_gap = percentile(chunk_gaps, 95)
    return output_text, usage, first_token_time, avg_gap, p95_gap, finish_reason


def stream_chat_completion(
    url: str,
    api_key: str,
    model: str,
    case: TestCase,
    counter: TokenCounter,
    timeout: int = 600,
    non_stream: bool = False,
) -> RunResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    base_payload: Dict[str, Any] = {
        "model": model,
        "messages": case.messages,
        "temperature": case.temperature,
        "top_p": case.top_p,
        "max_tokens": case.max_tokens,
        "presence_penalty": case.presence_penalty,
        "frequency_penalty": case.frequency_penalty,
    }

    start = time.perf_counter()
    status_code = 0

    try:
        if non_stream:
            payload = dict(base_payload)
            payload["stream"] = False
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            status_code = response.status_code
            headers_time_s = time.perf_counter() - start

            if response.status_code != 200:
                return RunResult(
                    test_name=case.name,
                    run_index=0,
                    status_code=status_code,
                    headers_time_s=headers_time_s,
                    ttft_s=None,
                    total_time_s=time.perf_counter() - start,
                    generation_time_s=None,
                    completion_tokens=None,
                    prompt_tokens=None,
                    total_tokens=None,
                    tok_per_sec=None,
                    ms_per_token=None,
                    avg_chunk_gap_s=None,
                    p95_chunk_gap_s=None,
                    finish_reason=None,
                    error=f"HTTP {status_code}: {response.text[:500]}",
                    output_text="",
                    token_count_source="none",
                )

            body = response.json()
            choice = (body.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "") or ""
            usage = body.get("usage") or {}
            end = time.perf_counter()
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            total_tokens = usage.get("total_tokens")
            tok_per_sec = None
            ms_per_token = None
            generation_time_s = None
            if completion_tokens and completion_tokens > 0:
                generation_time_s = end - start
                tok_per_sec = completion_tokens / generation_time_s
                ms_per_token = generation_time_s * 1000.0 / completion_tokens

            return RunResult(
                test_name=case.name,
                run_index=0,
                status_code=status_code,
                headers_time_s=headers_time_s,
                ttft_s=None,
                total_time_s=end - start,
                generation_time_s=generation_time_s,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                tok_per_sec=tok_per_sec,
                ms_per_token=ms_per_token,
                avg_chunk_gap_s=None,
                p95_chunk_gap_s=None,
                finish_reason=choice.get("finish_reason"),
                error=None,
                output_text=text,
                token_count_source="usage",
            )

        payload = dict(base_payload)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        status_code = response.status_code
        headers_time_s = time.perf_counter() - start

        if response.status_code != 200:
            return RunResult(
                test_name=case.name,
                run_index=0,
                status_code=status_code,
                headers_time_s=headers_time_s,
                ttft_s=None,
                total_time_s=time.perf_counter() - start,
                generation_time_s=None,
                completion_tokens=None,
                prompt_tokens=None,
                total_tokens=None,
                tok_per_sec=None,
                ms_per_token=None,
                avg_chunk_gap_s=None,
                p95_chunk_gap_s=None,
                finish_reason=None,
                error=f"HTTP {status_code}: {response.text[:500]}",
                output_text="",
                token_count_source="none",
            )

        output_text, usage, first_token_time, avg_gap, p95_gap, finish_reason = parse_openai_sse_lines(response)
        end = time.perf_counter()
        ttft_s = None if first_token_time is None else first_token_time - start
        generation_time_s = None if first_token_time is None else end - first_token_time

        prompt_tokens = usage.get("prompt_tokens") if usage else None
        completion_tokens = usage.get("completion_tokens") if usage else None
        total_tokens = usage.get("total_tokens") if usage else None
        token_count_source = "usage" if usage else "none"

        if completion_tokens is None or prompt_tokens is None or total_tokens is None:
            # Fallback counts when server does not emit usage in the stream.
            prompt_tokens = counter.count_messages(case.messages)
            completion_tokens = counter.count_text(output_text)
            total_tokens = prompt_tokens + completion_tokens
            token_count_source = counter.source if counter.source != "none" else "approx"

        tok_per_sec = None
        ms_per_token = None
        if generation_time_s is not None and completion_tokens and completion_tokens > 0:
            tok_per_sec = completion_tokens / generation_time_s
            ms_per_token = generation_time_s * 1000.0 / completion_tokens

        return RunResult(
            test_name=case.name,
            run_index=0,
            status_code=status_code,
            headers_time_s=headers_time_s,
            ttft_s=ttft_s,
            total_time_s=end - start,
            generation_time_s=generation_time_s,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            tok_per_sec=tok_per_sec,
            ms_per_token=ms_per_token,
            avg_chunk_gap_s=avg_gap,
            p95_chunk_gap_s=p95_gap,
            finish_reason=finish_reason,
            error=None,
            output_text=output_text,
            token_count_source=token_count_source,
        )

    except Exception as e:
        end = time.perf_counter()
        prompt_tokens = counter.count_messages(case.messages)
        completion_tokens = counter.count_text("")
        return RunResult(
            test_name=case.name,
            run_index=0,
            status_code=status_code,
            headers_time_s=end - start,
            ttft_s=None,
            total_time_s=end - start,
            generation_time_s=None,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_per_sec=None,
            ms_per_token=None,
            avg_chunk_gap_s=None,
            p95_chunk_gap_s=None,
            finish_reason=None,
            error=str(e),
            output_text="",
            token_count_source="approx",
        )


def print_run(result: RunResult, show_output: bool = False) -> None:
    print(f"\n=== {result.test_name} ===")
    print(f"Status            : {result.status_code}")
    print(f"Headers time      : {safe_float(result.headers_time_s)} s")
    print(f"TTFT              : {safe_float(result.ttft_s)} s")
    print(f"Total latency     : {safe_float(result.total_time_s)} s")
    print(f"Generation time   : {safe_float(result.generation_time_s)} s")
    print(f"Prompt tokens     : {safe_int(result.prompt_tokens)}")
    print(f"Completion tokens : {safe_int(result.completion_tokens)}")
    print(f"Total tokens      : {safe_int(result.total_tokens)}")
    print(f"Tokens/sec        : {safe_float(result.tok_per_sec)}")
    print(f"ms/token          : {safe_float(result.ms_per_token)}")
    print(f"Avg chunk gap     : {safe_float(result.avg_chunk_gap_s)} s")
    print(f"P95 chunk gap     : {safe_float(result.p95_chunk_gap_s)} s")
    print(f"Finish reason     : {result.finish_reason or '-'}")
    print(f"Token source      : {result.token_count_source}")
    print(f"Error             : {result.error or '-'}")
    if show_output:
        print("\nOutput:")
        print(result.output_text)
    else:
        preview = result.output_text[:500].replace("\n", "\\n")
        suffix = "..." if len(result.output_text) > 500 else ""
        print(f"Output preview    : {preview}{suffix}")


def summarize_results(results: List[RunResult]) -> None:
    by_test: Dict[str, List[RunResult]] = {}
    for r in results:
        by_test.setdefault(r.test_name, []).append(r)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    header = (
        f"{'Test':20}  {'Runs':>4}  {'TTFT avg':>10}  {'Total avg':>10}  "
        f"{'Tok/s avg':>10}  {'Comp avg':>10}  {'Errors':>6}  {'Source':>10}"
    )
    print(header)
    print("-" * len(header))

    all_ttft: List[float] = []
    all_total: List[float] = []
    all_tokps: List[float] = []

    for test_name, rows in by_test.items():
        ttfts = [r.ttft_s for r in rows if r.ttft_s is not None]
        totals = [r.total_time_s for r in rows if r.total_time_s is not None]
        tokps = [r.tok_per_sec for r in rows if r.tok_per_sec is not None]
        comp_tokens = [r.completion_tokens for r in rows if r.completion_tokens is not None]
        errors = sum(1 for r in rows if r.error)
        sources = {}
        for r in rows:
            sources[r.token_count_source] = sources.get(r.token_count_source, 0) + 1
        source_label = max(sources.items(), key=lambda x: x[1])[0] if sources else "-"
        comp_avg = mean_or_none([float(x) for x in comp_tokens]) if comp_tokens else None

        all_ttft.extend(ttfts)
        all_total.extend(totals)
        all_tokps.extend(tokps)

        print(
            f"{test_name:20}  {len(rows):>4}  {safe_float(mean_or_none(ttfts)):>10}  "
            f"{safe_float(mean_or_none(totals)):>10}  {safe_float(mean_or_none(tokps)):>10}  "
            f"{safe_float(comp_avg):>10}  {errors:>6}  {source_label:>10}"
        )

    print("-" * len(header))
    overall_completion = [float(r.completion_tokens) for r in results if r.completion_tokens is not None]
    print(
        f"{'OVERALL':20}  {len(results):>4}  {safe_float(mean_or_none(all_ttft)):>10}  "
        f"{safe_float(mean_or_none(all_total)):>10}  {safe_float(mean_or_none(all_tokps)):>10}  "
        f"{safe_float(mean_or_none(overall_completion)):>10}  {sum(1 for r in results if r.error):>6}  {'-':>10}"
    )

    if all_ttft:
        print("\nPercentiles (overall):")
        print(f"TTFT p50   : {safe_float(percentile(all_ttft, 50))} s")
        print(f"TTFT p95   : {safe_float(percentile(all_ttft, 95))} s")
    if all_total:
        print(f"Total p50  : {safe_float(percentile(all_total, 50))} s")
        print(f"Total p95  : {safe_float(percentile(all_total, 95))} s")
    if all_tokps:
        print(f"Tok/s p50  : {safe_float(percentile(all_tokps, 50))}")
        print(f"Tok/s p95  : {safe_float(percentile(all_tokps, 95))}")


def write_csv(path: str, results: List[RunResult]) -> None:
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible chat endpoint.")
    parser.add_argument("--url", default="https://kaggle-inference-proxy.onrender.com/v1/chat/completions")
    parser.add_argument("--api-key", required=True, help="API key / bearer token")
    parser.add_argument("--model", required=True, help="Model name to send in the request")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test case")
    parser.add_argument("--show-output", action="store_true", help="Print full output")
    parser.add_argument("--csv", default=None, help="Optional CSV path")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds")
    parser.add_argument("--long-context-repeat", type=int, default=1200)
    parser.add_argument("--non-stream", action="store_true", help="Disable streaming and rely on final usage")
    parser.add_argument("--hf-tokenizer", default=None, help="Optional Hugging Face tokenizer name for fallback counting")
    parser.add_argument("--tiktoken-model", default="cl100k_base", help="tiktoken model/encoding hint")
    args = parser.parse_args()

    tests = build_test_suite(long_context_repeat=args.long_context_repeat)
    counter = TokenCounter(args.hf_tokenizer, args.tiktoken_model)
    all_results: List[RunResult] = []

    print("=" * 100)
    print("Endpoint Benchmark")
    print(f"URL   : {args.url}")
    print(f"Model : {args.model}")
    print(f"Runs  : {args.runs} per test")
    print(f"Count : {counter.source}")
    print(f"Mode  : {'non-stream' if args.non_stream else 'stream + include_usage'}")
    print("=" * 100)

    for case in tests:
        for i in range(args.runs):
            print(f"\nRunning {case.name} [{i + 1}/{args.runs}]...")
            result = stream_chat_completion(
                url=args.url,
                api_key=args.api_key,
                model=args.model,
                case=case,
                counter=counter,
                timeout=args.timeout,
                non_stream=args.non_stream,
            )
            result.run_index = i + 1
            print_run(result, show_output=args.show_output)
            all_results.append(result)

    summarize_results(all_results)

    if args.csv:
        write_csv(args.csv, all_results)
        print(f"\nSaved CSV results to: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())