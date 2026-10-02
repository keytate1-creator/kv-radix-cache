"""
Integration tests for KVRadixCache lifecycle, LRU promotion, and memory budget eviction.
"""

import unittest
from radix_cache.cache import KVRadixCache


class TestKVRadixCache(unittest.TestCase):

    def test_cache_hit_and_stats(self):
        cache = KVRadixCache(max_tokens=1000)

        sys_prompt = (100, 200, 300)
        cache.insert(sys_prompt, kv_data="SYS_STATE")

        # Query 1: full hit on system prompt
        hit1 = cache.match_longest_prefix(sys_prompt)
        self.assertTrue(hit1.is_full_match)
        self.assertEqual(hit1.kv_data, "SYS_STATE")

        # Query 2: partial hit
        user_prompt = sys_prompt + (400, 500)
        hit2 = cache.match_longest_prefix(user_prompt)
        self.assertFalse(hit2.is_full_match)
        self.assertEqual(hit2.matched_tokens, sys_prompt)
        self.assertEqual(hit2.unmatched_tokens, (400, 500))

        # Query 3: total miss
        hit3 = cache.match_longest_prefix((999, 888))
        self.assertEqual(hit3.match_length, 0)

        stats = cache.get_stats()
        self.assertEqual(stats.total_queries, 3)
        self.assertEqual(stats.full_hits, 1)
        self.assertEqual(stats.partial_hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertAlmostEqual(stats.hit_rate, 2.0 / 3.0, places=4)

    def test_eviction_on_capacity(self):
        # Budget of 10 tokens
        cache = KVRadixCache(max_tokens=10)

        # Insert 6 tokens
        cache.insert((1, 2, 3, 4, 5, 6), kv_data="CHUNK_A")
        self.assertEqual(cache.tree.total_tokens, 6)

        # Insert another 6 tokens -> Total would be 12 > 10, triggers eviction of CHUNK_A
        cache.insert((10, 20, 30, 40, 50, 60), kv_data="CHUNK_B")

        self.assertLessEqual(cache.tree.total_tokens, 10)
        stats = cache.get_stats()
        self.assertGreaterEqual(stats.eviction_count, 1)

        # CHUNK_A was evicted
        match_a = cache.match_longest_prefix((1, 2, 3, 4, 5, 6))
        self.assertEqual(match_a.match_length, 0)

        # CHUNK_B remains
        match_b = cache.match_longest_prefix((10, 20, 30, 40, 50, 60))
        self.assertTrue(match_b.is_full_match)


if __name__ == "__main__":
    unittest.main()
