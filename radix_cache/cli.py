"""
Command-line interface and benchmark utility for KVRadixCache.
"""

from __future__ import annotations

import sys
import time
import random
import argparse
from typing import Sequence, Optional, List, Tuple

from radix_cache.cache import KVRadixCache


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kv-radix-cache",
        description="High-Performance Radix Prefix Trie & KV Cache Eviction Engine for LLM Serving.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: bench
    bench_parser = subparsers.add_parser("bench", help="Run prefix-sharing synthetic benchmark.")
    bench_parser.add_argument(
        "--num-prompts",
        type=int,
        default=2000,
        help="Number of simulated query prompts (default: 2000)",
    )
    bench_parser.add_argument(
        "--shared-prefix-len",
        type=int,
        default=64,
        help="Length of common system prompt token prefix (default: 64)",
    )
    bench_parser.add_argument(
        "--unique-suffix-len",
        type=int,
        default=32,
        help="Length of variable user prompt suffix (default: 32)",
    )
    bench_parser.add_argument(
        "--max-tokens",
        type=int,
        default=50000,
        help="Maximum cached token budget before eviction (default: 50000)",
    )
    bench_parser.add_argument(
        "--num-shared-prefixes",
        type=int,
        default=5,
        help="Number of distinct shared system prompts (default: 5)",
    )

    return parser


def generate_synthetic_workload(
    num_prompts: int,
    shared_prefixes: List[Tuple[int, ...]],
    suffix_len: int,
    seed: int = 42,
) -> List[Tuple[int, ...]]:
    rng = random.Random(seed)
    workload: List[Tuple[int, ...]] = []
    for _ in range(num_prompts):
        prefix = rng.choice(shared_prefixes)
        suffix = tuple(rng.randint(1000, 30000) for _ in range(suffix_len))
        workload.append(prefix + suffix)
    return workload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "bench":
        rng = random.Random(1337)
        shared_prefixes = [
            tuple(rng.randint(100, 999) for _ in range(args.shared_prefix_len))
            for _ in range(args.num_shared_prefixes)
        ]

        workload = generate_synthetic_workload(
            num_prompts=args.num_prompts,
            shared_prefixes=shared_prefixes,
            suffix_len=args.unique_suffix_len,
        )

        cache = KVRadixCache(max_tokens=args.max_tokens)

        # Pre-seed shared prefixes in cache
        for idx, prefix in enumerate(shared_prefixes):
            cache.insert(prefix, kv_data=f"BLOCK_SYSTEM_{idx}")

        start_time = time.perf_counter()

        for prompt in workload:
            match = cache.match_longest_prefix(prompt)
            if not match.is_full_match:
                cache.insert(prompt, kv_data="KV_NEW")

        elapsed_sec = time.perf_counter() - start_time
        stats = cache.get_stats()

        qps = args.num_prompts / elapsed_sec
        token_throughput = stats.tokens_queried / elapsed_sec

        sys.stdout.write("==================================================\n")
        sys.stdout.write(" KV RADIX CACHE BENCHMARK RESULTS\n")
        sys.stdout.write("==================================================\n")
        sys.stdout.write(f" Queries Processed   : {args.num_prompts:,}\n")
        sys.stdout.write(f" Elapsed Time        : {elapsed_sec:.4f} s\n")
        sys.stdout.write(f" Query Throughput    : {qps:,.0f} queries/sec\n")
        sys.stdout.write(f" Token Throughput    : {token_throughput:,.0f} tokens/sec\n")
        sys.stdout.write(f" Cache Hit Rate      : {stats.hit_rate:.2%}\n")
        sys.stdout.write(f" Token Reuse Rate    : {stats.token_reuse_rate:.2%}\n")
        sys.stdout.write(f" Total Tokens Cached : {stats.total_tokens_cached:,}\n")
        sys.stdout.write(f" Evictions Executed  : {stats.eviction_count:,}\n")
        sys.stdout.write("==================================================\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
