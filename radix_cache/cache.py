"""
Unified KVRadixCache manager combining the Radix Tree and LRU eviction engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from radix_cache.trie import RadixTree, RadixTrieNode, PrefixMatchResult
from radix_cache.eviction import DoublyLinkedList, LRUNode


@dataclass
class CacheStats:
    """Execution telemetry and cache performance statistics."""
    total_queries: int
    full_hits: int
    partial_hits: int
    misses: int
    tokens_queried: int
    tokens_reused: int
    total_tokens_cached: int
    eviction_count: int

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return (self.full_hits + self.partial_hits) / float(self.total_queries)

    @property
    def token_reuse_rate(self) -> float:
        if self.tokens_queried == 0:
            return 0.0
        return self.tokens_reused / float(self.tokens_queried)


class KVRadixCache:
    """
    In-memory Key-Value Radix Cache engine.
    Matches longest token prefixes in O(L) time and evicts unpinned leaf nodes in O(1).
    """

    def __init__(self, max_tokens: int = 100_000):
        if max_tokens < 1:
            raise ValueError("max_tokens budget must be >= 1")
        self.max_tokens = max_tokens
        self.tree: RadixTree = RadixTree()
        self.lru_list: DoublyLinkedList[RadixTrieNode] = DoublyLinkedList()
        # Map: node_id -> LRUNode
        self.lru_map: Dict[int, LRUNode[RadixTrieNode]] = {}

        # Telemetry
        self._total_queries: int = 0
        self._full_hits: int = 0
        self._partial_hits: int = 0
        self._misses: int = 0
        self._tokens_queried: int = 0
        self._tokens_reused: int = 0
        self._eviction_count: int = 0

    def match_longest_prefix(self, tokens: Tuple[int, ...]) -> PrefixMatchResult:
        """
        Queries the Radix Cache for the longest prefix match of the given token sequence.
        Updates LRU list and telemetry on hit.
        """
        self._total_queries += 1
        self._tokens_queried += len(tokens)

        result = self.tree.longest_prefix_match(tokens)

        if result.match_length > 0 and result.last_matched_node is not None:
            self._tokens_reused += result.match_length
            if result.is_full_match:
                self._full_hits += 1
            else:
                self._partial_hits += 1

            # Promote node in LRU list
            node_id = result.last_matched_node.node_id
            if node_id in self.lru_map:
                self.lru_list.move_to_front(self.lru_map[node_id])
        else:
            self._misses += 1

        return result

    def insert(self, tokens: Tuple[int, ...], kv_data: Any = None) -> RadixTrieNode:
        """
        Inserts token sequence and registers KV tensor pointers into cache.
        Enforces max_tokens memory limit by evicting unpinned leaves.
        """
        node = self.tree.insert(tokens, kv_data=kv_data)

        # Register in LRU
        if node.node_id not in self.lru_map:
            lru_entry = LRUNode(node.node_id, node)
            self.lru_map[node.node_id] = lru_entry
            self.lru_list.add_first(lru_entry)
        else:
            self.lru_list.move_to_front(self.lru_map[node.node_id])

        # Enforce memory budget
        self._evict_if_needed()

        return node

    def _evict_if_needed(self) -> None:
        """Evicts unpinned leaf nodes while cached token count exceeds max_tokens."""
        while self.tree.total_tokens > self.max_tokens and len(self.lru_list) > 0:
            candidate = self.lru_list.find_tail_candidate()
            if candidate is None:
                # All nodes are pinned, cannot evict further
                break

            trie_node = candidate.value

            if trie_node.is_leaf and not trie_node.is_root:
                # Successfully found an unpinned leaf
                node_id = candidate.key
                self.tree.remove_leaf(trie_node)
                self.lru_list.remove(candidate)
                del self.lru_map[node_id]
                self._eviction_count += 1
            else:
                # If candidate is an internal node (has children), cannot evict child-bearing node
                # Temporarily move to front or skip to avoid infinite loop
                self.lru_list.move_to_front(candidate)
                # Check if any leaves exist at all
                break

    def pin(self, node: RadixTrieNode) -> None:
        """Pins a node and its ancestors to guarantee they won't be evicted during generation."""
        curr: Optional[RadixTrieNode] = node
        while curr is not None and not curr.is_root:
            if curr.node_id in self.lru_map:
                self.lru_map[curr.node_id].pin()
            curr = curr.parent

    def unpin(self, node: RadixTrieNode) -> None:
        """Unpins a node and its ancestors."""
        curr: Optional[RadixTrieNode] = node
        while curr is not None and not curr.is_root:
            if curr.node_id in self.lru_map:
                self.lru_map[curr.node_id].unpin()
            curr = curr.parent

    def get_stats(self) -> CacheStats:
        """Returns consolidated telemetry snapshot."""
        return CacheStats(
            total_queries=self._total_queries,
            full_hits=self._full_hits,
            partial_hits=self._partial_hits,
            misses=self._misses,
            tokens_queried=self._tokens_queried,
            tokens_reused=self._tokens_reused,
            total_tokens_cached=self.tree.total_tokens,
            eviction_count=self._eviction_count,
        )
