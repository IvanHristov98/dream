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
    NodeID represents a NodeID. It could be used for comparison.
    """

    id: uuid.UUID

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "NodeID"):
        return self.id == other.id

    def __str__(self) -> str:
        return str(self.id)


class Node:
    id: NodeID
    is_root: bool
    children: Set[NodeID]
    # The coordinates of the node.
    vec: np.ndarray
    # Only leaf nodes have features.
    features: List[Feature]

    def __init__(self, id: NodeID, vec: np.ndarray, is_root: bool = False) -> None:
        self.id = id
        self.vec = vec
        self.children = set()
        self.features = list()
        self.is_root = is_root

    def add_child(self, child_id: NodeID) -> None:
        self.children.add(child_id)
