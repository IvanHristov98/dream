import abc
from typing import Callable

import dream.voctree.api as vtapi
from dream import model


class Store(vtapi.DocStore):
    def atomically(self, _: Callable[[any], None]) -> None:
        raise NotImplementedError("")

    def add_image_metadata(self, tx: any, im: model.Image) -> None:
        raise NotImplementedError("")


class ImageStore(abc.ABC):
    def store_matrix(self, _: model.Image) -> None:
        raise NotImplementedError("")


class SemSearchService:
    _vtree: vtapi.VocabularyTree
    _im_store: ImageStore
    _store: Store

    def __init__(self, vtree: vtapi.VocabularyTree, im_store: ImageStore, store: Store) -> None:
        self._vtree = vtree
        self._im_store = im_store
        self._store = store

    def seed_image(self, im: model.Image) -> None:
        self._im_store.store_matrix(im)

        # The code logic should always first search for image metadata in the db and then read the matrix.
        # Hence it is necessary to first write the matrix and then the metadata because they are operations
        # to different IO destinations.
        def _cb(tx: any) -> None:
            self._store.add_image_metadata(tx, im)

        self._store.atomically(_cb)
