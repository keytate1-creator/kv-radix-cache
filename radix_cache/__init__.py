"""
KVRadixCache: High-Performance Radix Prefix Trie & KV Cache Eviction Engine.
"""

from radix_cache.trie import RadixTrieNode, PrefixMatchResult
from radix_cache.eviction import DoublyLinkedList, LRUNode
from radix_cache.cache import KVRadixCache, CacheStats

__version__ = "0.1.0"
__author__ = "Kaylor Tate Cespedes"

__all__ = [
    "RadixTrieNode",
    "PrefixMatchResult",
    "DoublyLinkedList",
    "LRUNode",
    "KVRadixCache",
    "CacheStats",
]
