from typing import List
import abc

import cv2 as cv

import dream.voctree.api as vtapi
import dream.pg as dreampg
from dream import model
from dream.semsearch.docstore.im_metadata import sample_im_metadata


class MatrixLoader(abc.ABC):
    def load_matrix(self, _: model.Image) -> model.Image:
        raise NotImplementedError("")


class ImageStore(vtapi.DocStore):
    _SIFT_DESCRIPTOR_DIM = 128

    _sift: cv.SIFT
    _mat_loader: MatrixLoader

    def __init__(self, mat_loader) -> None:
        super().__init__()

        self._sift = cv.SIFT_create()
        self._mat_loader = mat_loader

    def get_documents(self, tx: any, sample_size: int) -> List[vtapi.Document]:
        pg_tx = dreampg.to_tx(tx)

        ims: List[model.Image] = sample_im_metadata(pg_tx)
        docs: List[vtapi.Document] = []

        for im in ims:
            im_with_mat = self._mat_loader.load_matrix(im)
            doc = self._im_to_doc(im_with_mat)

            docs.append(doc)

        return docs

    def _im_to_doc(self, im_with_mat: model.Image) -> vtapi.Document:
        mat_gray = cv.cvtColor(im_with_mat.mat, cv.COLOR_BGR2GRAY)

        (_, descriptor) = self._sift.detectAndCompute(mat_gray, None)

        vecs = list(descriptor)
        return vtapi.Document(id=im_with_mat.id, vectors=vecs)

    def get_feature_dim(self) -> int:
        return self._SIFT_DESCRIPTOR_DIM