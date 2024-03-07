import uuid
from typing import Set, List, NamedTuple

import numpy as np


class Feature(NamedTuple):
    doc_id: uuid.UUID
    vec: np.ndarray

    def __eq__(self, other: object) -> bool:
        if self.doc_id != other.doc_id:
            return False

        return np.allclose(self.vec, other.vec)


class Node:
    """
    Node is a node in a vocabulary tree.
    """

    id: uuid.UUID
    tree_id: uuid.UUID
    # depth of 0 means the root.
    depth: int
    # The coordinates of the node.
    vec: np.ndarray
    children: Set[uuid.UUID]
    # Only leaf nodes should have features.
    features: List[Feature]

    def __init__(self, id: uuid.UUID, tree_id: uuid.UUID, depth: int, vec: np.ndarray) -> None:
        self.id = id
        self.tree_id = tree_id
        self.depth = depth
        self.vec = vec

        self.children = set()
        self.features = list()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False

        if self.id != other.id or self.tree_id != other.tree_id or self.depth != other.depth:
            return False

        if not np.allclose(self.vec, other.vec):
            return False

        return self.children == other.children and self.features == other.features


class TrainJob:
    id: uuid.UUID
    added_node: Node

    def __init__(self, id: uuid.UUID, added_node: Node) -> None:
        self.id = id
        self.added_node = added_node

    def __eq__(self, other: object) -> bool:
        return self.id == other.id and self.added_node == other.added_node
