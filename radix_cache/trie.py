"""
Radix Tree (Prefix Trie) data structure with edge compression and longest prefix matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List


@dataclass
class PrefixMatchResult:
    """Carries the outcome of a longest prefix match traversal."""
    matched_tokens: Tuple[int, ...]
    unmatched_tokens: Tuple[int, ...]
    last_matched_node: Optional[RadixTrieNode]
    kv_data: Any = None

    @property
    def match_length(self) -> int:
        return len(self.matched_tokens)

    @property
    def is_full_match(self) -> bool:
        return len(self.unmatched_tokens) == 0


class RadixTrieNode:
    """
    An edge-compressed node in the Radix Trie.
    Edges represent sequences of token IDs (tuples of ints).
    """

    _id_counter: int = 0

    def __init__(
        self,
        token_slice: Tuple[int, ...] = (),
        parent: Optional[RadixTrieNode] = None,
        kv_data: Any = None,
    ):
        RadixTrieNode._id_counter += 1
        self.node_id: int = RadixTrieNode._id_counter
        self.token_slice: Tuple[int, ...] = token_slice
        self.parent: Optional[RadixTrieNode] = parent
        self.kv_data: Any = kv_data
        # Map: first token of child slice -> child node
        self.children: Dict[int, RadixTrieNode] = {}
        self.access_count: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def token_count(self) -> int:
        return len(self.token_slice)

    def total_tokens_in_path(self) -> int:
        """Computes total tokens along root-to-node path."""
        count = self.token_count
        curr = self.parent
        while curr is not None:
            count += curr.token_count
            curr = curr.parent
        return count

    def get_full_token_path(self) -> Tuple[int, ...]:
        """Constructs full token sequence from root down to this node."""
        path: List[Tuple[int, ...]] = []
        curr: Optional[RadixTrieNode] = self
        while curr is not None:
            if curr.token_slice:
                path.append(curr.token_slice)
            curr = curr.parent
        path.reverse()
        return tuple(tok for chunk in path for tok in chunk)


class RadixTree:
    """Radix Tree container orchestrating search, insertion, and edge splitting."""

    def __init__(self) -> None:
        self.root: RadixTrieNode = RadixTrieNode(token_slice=(), parent=None)
        self.total_nodes: int = 1
        self.total_tokens: int = 0

    def longest_prefix_match(self, tokens: Tuple[int, ...]) -> PrefixMatchResult:
        """
        Traverses tree matching token sequence against compressed edges.
        Returns longest matched prefix and corresponding node.
        """
        curr_node = self.root
        matched_tokens: List[int] = []
        remaining_tokens = tokens
        last_data_node: Optional[RadixTrieNode] = None

        while remaining_tokens:
            first_token = remaining_tokens[0]
            if first_token not in curr_node.children:
                break

            child = curr_node.children[first_token]
            edge = child.token_slice

            # Count common prefix between edge and remaining_tokens
            common_len = 0
            limit = min(len(edge), len(remaining_tokens))
            while common_len < limit and edge[common_len] == remaining_tokens[common_len]:
                common_len += 1

            if common_len == len(edge):
                # Full edge matched, advance to child
                matched_tokens.extend(edge)
                remaining_tokens = remaining_tokens[common_len:]
                curr_node = child
                if curr_node.kv_data is not None:
                    last_data_node = curr_node
            else:
                # Partial edge matched (diverges mid-edge)
                matched_tokens.extend(edge[:common_len])
                remaining_tokens = remaining_tokens[common_len:]
                # Cannot advance further into un-split child
                break

        final_data = last_data_node.kv_data if last_data_node else curr_node.kv_data

        return PrefixMatchResult(
            matched_tokens=tuple(matched_tokens),
            unmatched_tokens=remaining_tokens,
            last_matched_node=curr_node if curr_node != self.root else None,
            kv_data=final_data,
        )

    def insert(self, tokens: Tuple[int, ...], kv_data: Any = None) -> RadixTrieNode:
        """
        Inserts a token sequence into the Radix Tree.
        Performs edge splitting if a branching point occurs mid-edge.
        """
        if not tokens:
            self.root.kv_data = kv_data
            return self.root

        curr_node = self.root
        remaining_tokens = tokens

        while remaining_tokens:
            first_tok = remaining_tokens[0]

            if first_tok not in curr_node.children:
                # Direct insertion: create new leaf child
                new_leaf = RadixTrieNode(
                    token_slice=remaining_tokens,
                    parent=curr_node,
                    kv_data=kv_data,
                )
                curr_node.children[first_tok] = new_leaf
                self.total_nodes += 1
                self.total_tokens += len(remaining_tokens)
                return new_leaf

            child = curr_node.children[first_tok]
            edge = child.token_slice

            # Determine divergence point
            common_len = 0
            limit = min(len(edge), len(remaining_tokens))
            while common_len < limit and edge[common_len] == remaining_tokens[common_len]:
                common_len += 1

            if common_len == len(edge):
                # Complete edge matched: advance downwards
                remaining_tokens = remaining_tokens[common_len:]
                curr_node = child
                if not remaining_tokens:
                    # Exactly matches this node
                    curr_node.kv_data = kv_data
                    return curr_node
            else:
                # Mid-edge split required:
                # Split 'child' into:
                # 1. parent_split (common_len prefix)
                # 2. child_suffix (remaining edge of existing child)
                # 3. new_branch (remaining tokens of incoming sequence, if any)
                common_slice = edge[:common_len]
                existing_suffix = edge[common_len:]
                incoming_suffix = remaining_tokens[common_len:]

                # 1. Create intermediate node
                split_node = RadixTrieNode(
                    token_slice=common_slice,
                    parent=curr_node,
                    kv_data=None,
                )
                curr_node.children[first_tok] = split_node
                self.total_nodes += 1

                # 2. Re-attach existing child as child of split_node
                child.token_slice = existing_suffix
                child.parent = split_node
                split_node.children[existing_suffix[0]] = child

                # 3. Attach incoming branch if there are tokens left
                if incoming_suffix:
                    new_leaf = RadixTrieNode(
                        token_slice=incoming_suffix,
                        parent=split_node,
                        kv_data=kv_data,
                    )
                    split_node.children[incoming_suffix[0]] = new_leaf
                    self.total_nodes += 1
                    self.total_tokens += len(incoming_suffix)
                    return new_leaf
                else:
                    split_node.kv_data = kv_data
                    return split_node

        return curr_node

    def remove_leaf(self, leaf: RadixTrieNode) -> bool:
        """
        Removes a leaf node from the tree.
        Optionally merges parent edge if parent has only 1 remaining child and no data.
        Returns True if removed, False if node is not a leaf or is root.
        """
        if not leaf.is_leaf or leaf.is_root or leaf.parent is None:
            return False

        parent = leaf.parent
        first_tok = leaf.token_slice[0]

        if first_tok in parent.children and parent.children[first_tok] == leaf:
            del parent.children[first_tok]
            self.total_nodes -= 1
            self.total_tokens -= leaf.token_count

            # Edge merging optimization:
            # If parent is not root, has no kv_data of its own, and only 1 child remains:
            if not parent.is_root and parent.kv_data is None and len(parent.children) == 1:
                only_child = next(iter(parent.children.values()))
                # Merge parent with only_child
                parent.token_slice = parent.token_slice + only_child.token_slice
                parent.kv_data = only_child.kv_data
                parent.children = only_child.children
                for subchild in parent.children.values():
                    subchild.parent = parent
                self.total_nodes -= 1

            return True

        return False
