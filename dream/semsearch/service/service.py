import abc

import dream.voctree.api as vtapi
from dream import model


class Store(vtapi.DocStore):
    def add_image(self, tx: any) -> None:
        raise NotImplementedError("add_document is not implemented")


class ImageStore(abc.ABC):
    def store_matrix(self, _: model.Image) -> None:
        raise NotImplementedError("")

    def load_matrix(self, _: model.Image) -> model.Image:
        raise NotImplementedError("")


class SemSearchService:
    _vtree: vtapi.VocabularyTree

    def __init__(self, _vtree: vtapi.VocabularyTree) -> None:
        self._vtree = _vtree

    def seed_image(self, im: model.Image) -> bool:
        return False
