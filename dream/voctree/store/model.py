import uuid
from typing import List

import numpy as np

import dream.voctree.model as rsmodel


class PersistedFeature:
    doc_id: str
    vec: List[float]

    def __init__(self, doc_id: str, vec: List[float]) -> None:
        self.doc_id = doc_id
        self.vec = vec


def to_persisted_feature(feature: rsmodel.Feature) -> PersistedFeature:
    p_doc_id = feature.doc_id.hex
    p_vec = feature.vec.tolist()

    return PersistedFeature(p_doc_id, p_vec)


def to_feature(persisted: PersistedFeature) -> rsmodel.Feature:
    doc_id = uuid.UUID(persisted["doc_id"])
    vec = np.array(persisted["vec"], dtype=np.float32)

    return rsmodel.Feature(doc_id, vec)


class PersistedNode:
    id: uuid.UUID
    tree_id: uuid.UUID
    depth: int
    vec: List[float]
    children: List[str]
    features: List[PersistedFeature]

    def __init__(
        self, id: str, tree_id: str, depth: int, vec: List[float], children: List[str], features: List[PersistedFeature]
    ) -> None:
        self.id = id
        self.tree_id = tree_id
        self.depth = depth
        self.vec = vec
        self.children = children
        self.features = features


def to_persisted_node(node: rsmodel.Node) -> PersistedNode:
    p_node_id = node.id
    p_tree_id = node.tree_id
    p_vec = node.vec.tolist()

    p_children = []

    for child_id in node.children:
        p_children.append(child_id.hex)

    p_features = []

    for feature in node.features:
        p_feature = to_persisted_feature(feature)
        p_features.append(p_feature)

    return PersistedNode(p_node_id, p_tree_id, node.depth, p_vec, p_children, p_features)


def to_node(p_node: PersistedNode) -> rsmodel.Node:
    vec = np.array(p_node.vec, dtype=np.float32)

    node = rsmodel.Node(p_node.id, p_node.tree_id, p_node.depth, vec)

    for p_child_id in p_node.children:
        node.children.add(uuid.UUID(p_child_id))

    for p_feature in p_node.features:
        feature = to_feature(p_feature)
        node.features.append(feature)

    return node
