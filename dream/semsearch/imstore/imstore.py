from pathlib import Path

import cv2 as cv

import dream.semsearch.service as semsearchservice
from dream import model


_JPG_EXT = "jpg"


class ImageStore(semsearchservice.ImageStore):
    _ims_path: Path

    def __init__(self, ims_path: Path) -> None:
        super().__init__()

        self._ims_path = ims_path

    def store_matrix(self, im: model.Image) -> None:
        im_path = self._im_path(im.id)
        cv.imwrite(str(im_path.resolve()), im.mat)

    def load_matrix(self, im: model.Image) -> model.Image:
        im_path = self._im_path(im.id)
        mat = cv.imread(str(im_path))

        return im.with_mat(mat)

    def _im_path(self, im_id: model.ImageID) -> Path:
        return Path(self._ims_path, f"{str(im_id)}.{_JPG_EXT}")
