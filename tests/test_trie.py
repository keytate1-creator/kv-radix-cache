"""
Unit tests for RadixTree insertion, edge splitting, and longest prefix matching.
"""

import unittest
from radix_cache.trie import RadixTree


class TestRadixTree(unittest.TestCase):

    def test_insert_and_exact_match(self):
        tree = RadixTree()
        tokens = (10, 20, 30, 40)
        node = tree.insert(tokens, kv_data="KV_1")

        self.assertEqual(node.kv_data, "KV_1")
        self.assertEqual(node.token_slice, tokens)

        match = tree.longest_prefix_match(tokens)
        self.assertTrue(match.is_full_match)
        self.assertEqual(match.matched_tokens, tokens)
        self.assertEqual(match.unmatched_tokens, ())
        self.assertEqual(match.kv_data, "KV_1")

    def test_partial_prefix_match(self):
        tree = RadixTree()
        tree.insert((1, 2, 3, 4), kv_data="DATA_A")

        # Query with extra tokens
        query = (1, 2, 3, 4, 5, 6)
        match = tree.longest_prefix_match(query)

        self.assertFalse(match.is_full_match)
        self.assertEqual(match.matched_tokens, (1, 2, 3, 4))
        self.assertEqual(match.unmatched_tokens, (5, 6))
        self.assertEqual(match.kv_data, "DATA_A")

    def test_edge_splitting(self):
        tree = RadixTree()
        # Insert (1, 2, 3, 4, 5)
        tree.insert((1, 2, 3, 4, 5), kv_data="FULL_1")

        # Insert (1, 2, 3, 8, 9) -> Diverges at index 3
        tree.insert((1, 2, 3, 8, 9), kv_data="FULL_2")

        # The root should have 1 child starting with token 1: the split node (1, 2, 3)
        self.assertIn(1, tree.root.children)
        split_node = tree.root.children[1]
        self.assertEqual(split_node.token_slice, (1, 2, 3))
        self.assertEqual(len(split_node.children), 2)

        # Children of split node should start with 4 and 8
        self.assertIn(4, split_node.children)
        self.assertIn(8, split_node.children)
        self.assertEqual(split_node.children[4].token_slice, (4, 5))
        self.assertEqual(split_node.children[8].token_slice, (8, 9))

        # Query both
        match1 = tree.longest_prefix_match((1, 2, 3, 4, 5))
        self.assertTrue(match1.is_full_match)
        self.assertEqual(match1.kv_data, "FULL_1")

        match2 = tree.longest_prefix_match((1, 2, 3, 8, 9))
        self.assertTrue(match2.is_full_match)
        self.assertEqual(match2.kv_data, "FULL_2")

        # Query only the common prefix
        match_prefix = tree.longest_prefix_match((1, 2, 3, 99))
        self.assertEqual(match_prefix.matched_tokens, (1, 2, 3))
        self.assertEqual(match_prefix.unmatched_tokens, (99,))


if __name__ == "__main__":
    unittest.main()
