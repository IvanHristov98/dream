from pathlib import Path
import glob
from typing import List

import cv2 as cv

from dream.revsearch import service
from dream import model


_JPG_EXT = "jpg"


class FSImageStore(service.ImageStore):
    _ims_path: Path

    def __init__(self, ims_path: Path) -> None:
        super().__init__()

        self._ims_path = ims_path

    def store_matrix(self, im: model.Image) -> None:
        im_path = Path(self._ims_path, f"{str(im.id)}.{_JPG_EXT}")
        cv.imwrite(str(im_path.resolve()), im.mat)

    def load_matrix(self, im: model.Image) -> model.Image:
        pass
