from pathlib import Path
import uuid

import cv2 as cv

import dream.semsearch.service as semsearchservice
from dream import model
from dream.semsearch.docstore import image as imdocstore


_JPG_EXT = "jpg"


class ImageStore(semsearchservice.ImageStore):
    _ims_path: Path

    def __init__(self, ims_path: Path) -> None:
        super().__init__()

        self._ims_path = ims_path

    def store_matrix(self, im: model.Image) -> None:
        im_path = _im_path(self._ims_path, im.id)
        cv.imwrite(str(im_path.resolve()), im.mat)


class MatrixLoader(imdocstore.MatrixLoader):
    _ims_path: Path

    def __init__(self, ims_path: Path) -> None:
        super().__init__()

        self._ims_path = ims_path

    def load_matrix(self, im: model.Image) -> model.Image:
        im_path = _im_path(self._ims_path, im.id)
        mat = cv.imread(str(im_path))

        return im.with_mat(mat)


def _im_path(ims_path: Path, im_id: uuid.UUID) -> Path:
    return Path(ims_path, f"{str(im_id)}.{_JPG_EXT}")
