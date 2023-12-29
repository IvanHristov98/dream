import uuid
from typing import Set, List, NamedTuple

import numpy as np


class Document(NamedTuple):
    """
    Document may be any entity that can be semantically represented with a set descriptor vectors.
    Examples are images, sentences, etc.
    """

    id: uuid.UUID
    vectors: List[np.ndarray]


class Feature(NamedTuple):
    doc_id: uuid.UUID
    vec: np.ndarray


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


class NodeAdded:
    id: uuid.UUID
    node: Node
