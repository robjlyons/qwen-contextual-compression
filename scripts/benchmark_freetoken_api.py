"""Deterministic request-level benchmark for stock and QCC-patched FreeToken servers."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FreeToken's OpenAI-compatible API; elapsed time includes TTFT.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, default=Path("configs/freetoken_benchmark_prompts.json"))
    parser.add_argument("--runs", type=int, default=1, help="Number of complete passes over the prompt file")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def request_once(endpoint: str, model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}).encode()
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"FreeToken API returned HTTP {error.code}: {detail}") from error
    elapsed = time.perf_counter() - started
    choice = body["choices"][0]
    text = choice.get("message", {}).get("content", "")
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    return {
        "elapsed_seconds": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / elapsed if completion_tokens is not None else None,
        "finish_reason": choice.get("finish_reason"),
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "response_text": text,
        "prompt": prompt,
    }


def main():
    args = parse_args()
    if args.runs < 1 or args.warmup_runs < 0:
        raise ValueError("--runs must be positive and --warmup-runs non-negative")
    prompts = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(item, str) for item in prompts):
        raise ValueError("prompt file must contain a non-empty JSON array of strings")
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    for _ in range(args.warmup_runs):
        for prompt in prompts:
            request_once(endpoint, args.model, prompt, args.max_tokens, args.temperature)
    records = []
    for run in range(args.runs):
        for prompt_index, prompt in enumerate(prompts):
            records.append({"run": run, "prompt_index": prompt_index, **request_once(endpoint, args.model, prompt, args.max_tokens, args.temperature)})
    elapsed = [item["elapsed_seconds"] for item in records]
    rates = [item["completion_tokens_per_second"] for item in records if item["completion_tokens_per_second"] is not None]
    summary = {
        "measurement": "request-level throughput including TTFT; not pure decode-kernel tokens/s",
        "requests": len(records),
        "mean_elapsed_seconds": statistics.fmean(elapsed),
        "median_elapsed_seconds": statistics.median(elapsed),
        "mean_completion_tokens_per_second": statistics.fmean(rates) if rates else None,
        "median_completion_tokens_per_second": statistics.median(rates) if rates else None,
        "total_completion_tokens": sum(item["completion_tokens"] or 0 for item in records),
    }
    result = {"config": vars(args) | {"prompt_file": str(args.prompt_file), "output": str(args.output)}, "summary": summary, "requests": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
