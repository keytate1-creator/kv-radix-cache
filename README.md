# KVRadixCache: High-Performance Radix Prefix Trie & KV Cache Eviction Engine for LLM Serving

[![CI](https://github.com/keytate1-creator/kv-radix-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/keytate1-creator/kv-radix-cache/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Flake8](https://img.shields.io/badge/code%20style-flake8-000000.svg)](https://flake8.pycqa.org/)

`KVRadixCache` is a zero-dependency, production-grade memory management engine implementing **Radix Tree (Prefix Trie) Key-Value Cache reuse and eviction** for Large Language Model inference runtimes (e.g., vLLM, SGLang, TensorRT-LLM).

In multi-turn conversations, agentic workflows, and shared system prompts, up to **80% of prompt tokens are redundant**. By indexing prompt token IDs in an edge-compressed Radix Tree paired with an $O(1)$ LRU doubly-linked list eviction structure, `KVRadixCache` allows inference engines to match arbitrary-length shared prefixes in $O(L)$ time and reclaim GPU/Host memory under strict memory limits.

---

## Architectural Specification

```
                               Incoming Prompt Token Stream
                                 [101, 2045, 1037, 4522, 102]
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │     RadixTrie Search        │
                               │  (Longest Prefix Matching)  │
                               └──────────────┬──────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
          Full or Partial Cache Hit                            Cache Miss
      Prefix [101, 2045, 1037] Reused                     Insert New Node Chunk
       - Bump LRU Priority                                 - Split Edges if Needed
       - Return Cached KV Block Pointers                   - Allocate Memory Budget
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   Memory Eviction Manager   │
                               │   (O(1) Doubly-Linked LRU)  │
                               │  - Prune Lowest-Rank Leaves │
                               │  - Reclaim Capacity Budget  │
                               └─────────────────────────────┘
```

---

## Theoretical Properties & Invariants

1. **Edge Compression:** Shared token sequences are collapsed into single contiguous node edges to prevent tree bloat. A node is only split when two sequences diverge at offset $k$.
2. **Deterministic Longest Prefix Matching (LPM):** Given token query $Q$, traverses the tree matching common sub-arrays until the first mismatch. Returns $(K_{\text{matched}}, V_{\text{cached}}, Q_{\text{remaining}})$.
3. **Safe Leaf-First Eviction:** A node can only be evicted if it has **zero active children** (leaf node) and is not pinned by an active generation request.
4. **$O(1)$ LRU Maintenance:** Every match or insertion promotes the associated node to the head of a doubly-linked list. Eviction pulls strictly from the tail.

---

## Installation

```bash
git clone https://github.com/keytate1-creator/kv-radix-cache.git
cd kv-radix-cache
pip install -e .
```

---

## Programmatic Usage

```python
from radix_cache.cache import KVRadixCache

# Initialize cache with a budget of 10,000 cached tokens
cache = KVRadixCache(max_tokens=10000)

# System prompt tokens shared across requests
system_prompt = (101, 2054, 1037, 3422, 102)
agent_a_prompt = system_prompt + (4500, 2001, 102)
agent_b_prompt = system_prompt + (4500, 2005, 102)

# Insert Agent A's precomputed KV state
cache.insert(agent_a_prompt, kv_data="GPU_BLOCKS_A")

# Match Agent B: Shares system_prompt + token 4500 (length 6)
hit = cache.match_longest_prefix(agent_b_prompt)

print(f"Matched Prefix Tokens: {hit.matched_tokens}")
print(f"Matched Token Count  : {hit.match_length} / {len(agent_b_prompt)}")
print(f"Remaining Uncached   : {hit.unmatched_tokens}")
print(f"Cached State Data    : {hit.kv_data}")

# Check global cache metrics
stats = cache.get_stats()
print(f"Cache Hit Rate: {stats.hit_rate:.2%}")
print(f"Total Cached Tokens: {stats.total_tokens_cached}")
```

---

## CLI Benchmark

Benchmark cache insertion, prefix matching, and leaf eviction directly:

```bash
python -m radix_cache.cli bench --num-prompts 5000 --shared-prefix-len 128 --max-tokens 50000
```

---

## License

MIT License. Designed and maintained by Kaylor Tate Cespedes.
