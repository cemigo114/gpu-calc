#!/usr/bin/env python3
"""Convert SemiAnalysis Weka CC traces to Mooncake benchmark replay format.

Source: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-061526
Target: JSONL format compatible with llm-d benchmark_stage.py

Usage:
    # Download the dataset first:
    # huggingface-cli download semianalysisai/cc-traces-weka-061526 --repo-type dataset
    #
    # Or download the parquet directly:
    # wget https://huggingface.co/datasets/semianalysisai/cc-traces-weka-061526/resolve/main/data/train-00000-of-00001.parquet

    python3 weka_to_mooncake.py --input traces.json --output weka_agentic.jsonl
    python3 weka_to_mooncake.py --input traces.json --output weka_agentic.jsonl --max-input-tokens 8192
    python3 weka_to_mooncake.py --input traces.json --output weka_agentic.jsonl --max-sessions 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def convert_session(
    session: dict[str, Any],
    max_input_tokens: int,
    target_model: str,
) -> list[dict[str, Any]]:
    """Convert a single Weka session to Mooncake trace requests.

    Weka format per request:
        t: relative timestamp (seconds from session start)
        model: Claude model name
        in: input tokens (multiple of block_size=64)
        out: output tokens
        hash_ids: list of 64-token prefix block IDs
        api_time: API response latency
        type: request type

    Mooncake format per request:
        timestamp: relative timestamp
        input_tokens: input token count
        output_tokens: output token count
        num_hash_ids: number of prefix blocks (for cache simulation)
    """
    requests = session.get("requests", [])
    if not requests:
        return []

    converted = []
    for req in requests:
        if req.get("type") not in ("s", "n"):
            continue

        input_tokens = req.get("in", 0)
        if input_tokens == 0:
            continue
        if input_tokens > max_input_tokens:
            input_tokens = max_input_tokens

        output_tokens = req.get("out", 0)
        if output_tokens < 1:
            output_tokens = 1

        hash_ids = req.get("hash_ids", [])

        converted.append({
            "timestamp": req.get("t", 0),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_hash_ids": len(hash_ids),
            "model": target_model,
        })

    return converted


def compute_stats(all_requests: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for the converted trace."""
    if not all_requests:
        return {}

    input_tokens = [r["input_tokens"] for r in all_requests]
    output_tokens = [r["output_tokens"] for r in all_requests]

    return {
        "total_requests": len(all_requests),
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "avg_input_tokens": sum(input_tokens) // len(input_tokens),
        "avg_output_tokens": sum(output_tokens) // len(output_tokens),
        "max_input_tokens": max(input_tokens),
        "max_output_tokens": max(output_tokens),
        "avg_hash_ids": sum(r["num_hash_ids"] for r in all_requests) // len(all_requests),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SemiAnalysis Weka CC traces to Mooncake replay format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input JSON file (Weka trace format)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSONL file (Mooncake trace format)"
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=4096,
        help="Maximum input tokens per request (default: 4096)"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=0,
        help="Maximum sessions to convert (0 = all)"
    )
    parser.add_argument(
        "--max-requests", type=int, default=500,
        help="Maximum total requests in output (default: 500)"
    )
    parser.add_argument(
        "--target-model", default="glm-5.2-fp8",
        help="Target model name for replay (default: glm-5.2-fp8)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path}...")

    if input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
        sessions = data if isinstance(data, list) else [data]
    elif input_path.suffix == ".jsonl":
        sessions = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sessions.append(json.loads(line))
    else:
        print(f"Error: Unsupported format: {input_path.suffix} (use .json or .jsonl)", file=sys.stderr)
        sys.exit(1)

    if args.max_sessions > 0:
        sessions = sessions[:args.max_sessions]

    print(f"Converting {len(sessions)} sessions (max_input_tokens={args.max_input_tokens})...")

    all_requests: list[dict] = []
    for session in sessions:
        converted = convert_session(session, args.max_input_tokens, args.target_model)
        all_requests.extend(converted)
        if args.max_requests > 0 and len(all_requests) >= args.max_requests:
            all_requests = all_requests[:args.max_requests]
            break

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for req in all_requests:
            f.write(json.dumps(req) + "\n")

    stats = compute_stats(all_requests)
    print(f"\nWrote {len(all_requests)} requests to {output_path}")
    print(f"Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
