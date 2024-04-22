import abc
from typing import Callable, List
import uuid

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


class DocumentFactory(abc.ABC):
    def from_caption(self, doc_id: uuid.UUID, captions: List[str]) -> vtapi.Document:
        raise NotImplementedError("")


class SemSearchService:
    _captions_vtree: vtapi.VocabularyTree
    _ims_vtree: vtapi.VocabularyTree
    _im_store: ImageStore
    _store: Store
    _doc_factory: DocumentFactory

    def __init__(
        self,
        captions_vtree: vtapi.VocabularyTree,
        ims_vtree: vtapi.VocabularyTree,
        im_store: ImageStore,
        store: Store,
        doc_factory: DocumentFactory,
    ) -> None:
        self._captions_vtree = captions_vtree
        self._ims_vtree = ims_vtree
        self._im_store = im_store
        self._store = store
        self._doc_factory = doc_factory

    def seed_image(self, im: model.Image) -> None:
        self._im_store.store_matrix(im)

        # The code logic should always first search for image metadata in the db and then read the matrix.
        # Hence it is necessary to first write the matrix and then the metadata because they are operations
        # to different IO destinations.
        def _cb(tx: any) -> None:
            self._store.add_image_metadata(tx, im)

        self._store.atomically(_cb)

    def query(self, caption: str, n: int) -> List[uuid.UUID]:
        doc = self._doc_factory.from_caption(doc_id=uuid.uuid4(), captions=[caption])
        doc_ids = self._captions_vtree.query(doc, n)

        return doc_ids
