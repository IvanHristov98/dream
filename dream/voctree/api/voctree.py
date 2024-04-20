from abc import ABC
from typing import List
import uuid

import numpy as np


class ErrNodeNotFound(Exception):
    """
    ErrNodeNotFound is thrown whenever a node is not found.
    """


class ErrTrainingInProgress(Exception):
    """
    ErrTrainingInProgress is thrown when tree is added while training of another is in progress.
    """


class Document:
    """
    Document may be any entity that can be semantically represented with a set descriptor vectors.
    Examples are images, sentences, etc.
    """

    id: uuid.UUID
    vectors: List[np.ndarray]

    def __init__(self, doc_id: uuid.UUID, vectors: List[np.ndarray]) -> None:
        self.id = doc_id
        self.vectors = vectors


class DocStore(ABC):
    def sample_next_documents(self, tx: any, sample_size: int, tree_id: uuid.UUID) -> List[Document]:
        """
        sample_next_documents samples a set of documents for a given tree.
        If documents are previously sampled for a given tree, then they won't be sampled for it again.
        """
        raise NotImplementedError("")

    def get_feature_dim(self) -> int:
        raise NotImplementedError("")


class VocabularyTree(ABC):
    def start_training(self, sample_size: int) -> None:
        raise NotImplementedError("start_training is not implemented")

    def try_training(self) -> bool:
        raise NotImplementedError("try_training is not implemented")

    def try_replace_blue_tree(self) -> None:
        raise NotImplementedError("try_training is not implemented")

    def query(self, doc: Document, n: int) -> List[uuid.UUID]:
        raise NotImplementedError("query is not implemented")
