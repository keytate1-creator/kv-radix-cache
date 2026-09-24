"""
O(1) Doubly-Linked List and LRU eviction primitives for Radix Cache nodes.
"""

from __future__ import annotations

from typing import Generic, TypeVar, Optional

T = TypeVar("T")


class LRUNode(Generic[T]):
    """Node element within the LRU doubly-linked list."""

    def __init__(self, key: int, value: T):
        self.key: int = key
        self.value: T = value
        self.prev: Optional[LRUNode[T]] = None
        self.next: Optional[LRUNode[T]] = None
        self.pinned: bool = False

    def pin(self) -> None:
        """Pins the node to prevent eviction while active generation uses it."""
        self.pinned = True

    def unpin(self) -> None:
        """Unpins the node, making it eligible for eviction if it becomes a leaf."""
        self.pinned = False


class DoublyLinkedList(Generic[T]):
    """Sentinel-bounded doubly linked list supporting O(1) removals and promotions."""

    def __init__(self) -> None:
        # Dummy sentinels
        self.head: LRUNode[T] = LRUNode(-1, None)  # type: ignore
        self.tail: LRUNode[T] = LRUNode(-2, None)  # type: ignore
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def add_first(self, node: LRUNode[T]) -> None:
        """Inserts node at the head (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node
        self._size += 1

    def remove(self, node: LRUNode[T]) -> None:
        """Removes an arbitrary node from the list in O(1)."""
        if node.prev is not None and node.next is not None:
            node.prev.next = node.next
            node.next.prev = node.prev
            node.prev = None
            node.next = None
            self._size -= 1

    def move_to_front(self, node: LRUNode[T]) -> None:
        """Promotes node to head as the most recently accessed item."""
        self.remove(node)
        self.add_first(node)

    def find_tail_candidate(self) -> Optional[LRUNode[T]]:
        """
        Scans from least recently used (tail) towards head to find
        the first unpinned node.
        """
        curr = self.tail.prev
        while curr and curr != self.head:
            if not curr.pinned:
                return curr
            curr = curr.prev
        return None
