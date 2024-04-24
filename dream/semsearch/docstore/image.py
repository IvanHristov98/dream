from typing import List
import abc
import uuid

import cv2 as cv
import numpy as np

import dream.voctree.api as vtapi
import dream.pg as dreampg
from dream import model
from dream.semsearch.docstore.im_metadata import sample_im_metadata
from dream.semsearch import service


_SIFT_DESCRIPTOR_DIM = 128
_MAX_FEATURES_PER_IM = 150


class MatrixLoader(abc.ABC):
    def load_matrix(self, _: model.Image) -> model.Image:
        raise NotImplementedError("")


class ImageStore(vtapi.DocStore):
    _DOC_STORE = "images"

    _sift: cv.SIFT
    _mat_loader: MatrixLoader

    def __init__(self, mat_loader: MatrixLoader) -> None:
        super().__init__()

        self._sift = _new_sift()
        self._mat_loader = mat_loader

    def sample_next_documents(self, tx: any, sample_size: int, tree_id: uuid.UUID) -> List[vtapi.Document]:
        pg_tx = dreampg.to_tx(tx)

        ims: List[model.Image] = sample_im_metadata(pg_tx, sample_size, tree_id, self._DOC_STORE)
        docs: List[vtapi.Document] = []

        for im in ims:
            im_with_mat = self._mat_loader.load_matrix(im)
            doc = self._im_to_doc(im_with_mat)

            docs.append(doc)

        return docs

    def _im_to_doc(self, im_with_mat: model.Image) -> vtapi.Document:
        vecs = _extract_vecs(self._sift, im_with_mat.mat)
        return vtapi.Document(doc_id=im_with_mat.id, vectors=vecs)

    def get_feature_dim(self) -> int:
        return _SIFT_DESCRIPTOR_DIM


class ImageFeatureExtractor(service.ImageFeatureExtractor):
    _sift: cv.SIFT

    def __init__(self) -> None:
        super().__init__()

        self._sift = _new_sift()

    def extract(self, mat: np.ndarray) -> List[np.ndarray]:
        return _extract_vecs(self._sift, mat)


def _new_sift() -> cv.SIFT:
    return cv.SIFT_create(_MAX_FEATURES_PER_IM)


def _extract_vecs(sift: cv.SIFT, mat: np.ndarray) -> List[np.ndarray]:
    mat_gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)
    (_, descriptor) = sift.detectAndCompute(mat_gray, None)

    if descriptor is None:
        return []

    return list(descriptor)
