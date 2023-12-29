from typing import List, NamedTuple, Set
import uuid

import numpy as np

from dream.model import ImageID


class Feature(NamedTuple):
    """
    Feature is used to perform a reverse lookup from a feature to its image.
    """

    vec: np.ndarray
    im_id: ImageID


class NodeID(NamedTuple):
    """
    NodeID represents a node ID. It could be used for comparison.
    """

    id: uuid.UUID

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "NodeID"):
        return self.id == other.id

    def __str__(self) -> str:
        return str(self.id)


class Node:
    """
    Node is a node in a vocabulary tree.
    """

    id: NodeID
    tree_id: uuid.UUID
    # depth of 0 means the root.
    depth: int
    # The coordinates of the node.
    vec: np.ndarray
    # im_count is the number of unique images with at least one descriptor vector
    # through the node. Used for fast computation of IDF.
    im_count: int
    children: Set[NodeID]
    # Only leaf nodes should have features.
    features: List[Feature]

    def __init__(self, id: NodeID, tree_id: uuid.UUID, depth: int, vec: np.ndarray, im_count: int) -> None:
        self.id = id
        self.tree_id = tree_id
        self.depth = depth
        self.vec = vec
        self.im_count = im_count

        self.children = set()
        self.features = list()

    def add_child(self, child_id: NodeID) -> None:
        self.children.add(child_id)


class NodeAdded:
    id: uuid.UUID
    node_id: NodeID
