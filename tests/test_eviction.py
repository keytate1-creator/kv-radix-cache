"""
Unit tests for DoublyLinkedList and LRU node management using unittest.
"""

import unittest
from radix_cache.eviction import DoublyLinkedList, LRUNode


class TestEviction(unittest.TestCase):

    def test_doubly_linked_list_push_remove(self):
        dll = DoublyLinkedList[str]()
        self.assertTrue(dll.is_empty())
        self.assertEqual(len(dll), 0)

        n1 = LRUNode(1, "A")
        n2 = LRUNode(2, "B")
        n3 = LRUNode(3, "C")

        dll.add_first(n1)
        dll.add_first(n2)
        dll.add_first(n3)

        self.assertEqual(len(dll), 3)
        self.assertEqual(dll.head.next, n3)
        self.assertEqual(dll.tail.prev, n1)

        # Move n1 to front
        dll.move_to_front(n1)
        self.assertEqual(dll.head.next, n1)
        self.assertEqual(dll.tail.prev, n2)

        # Remove n2
        dll.remove(n2)
        self.assertEqual(len(dll), 2)
        self.assertEqual(dll.tail.prev, n3)

    def test_pinning_and_candidate_eviction(self):
        dll = DoublyLinkedList[str]()
        n1 = LRUNode(1, "A")
        n2 = LRUNode(2, "B")

        dll.add_first(n1)
        dll.add_first(n2)

        # Initially, tail candidate is n1 (least recently used)
        candidate = dll.find_tail_candidate()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.key, 1)

        # Pin n1
        n1.pin()
        # Candidate should now be n2
        candidate = dll.find_tail_candidate()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.key, 2)

        # Pin n2 as well
        n2.pin()
        # No candidate available
        self.assertIsNone(dll.find_tail_candidate())

        # Unpin n1
        n1.unpin()
        self.assertEqual(dll.find_tail_candidate().key, 1)


if __name__ == "__main__":
    unittest.main()
