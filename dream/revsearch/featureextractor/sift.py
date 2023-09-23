from typing import List

import cv2 as cv

from dream.revsearch import service
from dream import model
import dream.revsearch.model as rsmodel


_SIFT_DESCRIPTOR_DIM = 128


class SiftExtractor(service.FeatureExtractor):
    _sift: cv.SIFT

    def __init__(self) -> None:
        super().__init__()

        self._sift = cv.SIFT_create()

    def features(self, im: model.Image) -> List[rsmodel.Feature]:
        mat_gray = cv.cvtColor(im.mat, cv.COLOR_BGR2GRAY)

        (_, descriptor) = self._sift.detectAndCompute(mat_gray, None)

        return list(descriptor)

    def dim(self) -> int:
        return _SIFT_DESCRIPTOR_DIM
