import json
from pathlib import Path
from typing import Dict, List
import uuid

import cv2 as cv

from dream.dataset import dataset
from dream import model


class ImageMetadata:
    id: int
    file_name: str
    captions: List[str]

    def __init__(self, im_id: int, file_name: str) -> None:
        self.id = im_id
        self.file_name = file_name
        self.captions = []

    def add_caption(self, caption: str) -> None:
        self.captions.append(caption)


class COCO2014Iterator(dataset.DatasetIterator):
    _DATASET_NAME = "coco2014"

    _ims_path: Path
    _captioned_ims: List[ImageMetadata]
    _index: int

    def __init__(self, captions_path: Path, ims_path: Path) -> None:
        super().__init__()

        self._ims_path = ims_path

        with open(captions_path, encoding="utf-8") as captions_file:
            captions = json.load(captions_file)

        captioned_ims: Dict[int, ImageMetadata] = {}

        for image in captions["images"]:
            captioned_ims[image["id"]] = ImageMetadata(image["id"], image["file_name"])

        for annotation in captions["annotations"]:
            captioned_ims[annotation["image_id"]].add_caption(annotation["caption"])

        self._captioned_ims = list(captioned_ims.values())
        self._index = -1

    def next(self) -> bool:
        if self._index + 1 < len(self._captioned_ims):
            self._index += 1
            return True

        return False

    def read(self) -> model.Image:
        im_metadata = self._captioned_ims[self._index]

        im_path = Path(self._ims_path, im_metadata.file_name)
        mat = cv.imread(str(im_path))

        return (
            model.Image()
            .with_id(uuid.uuid4())
            .with_mat(mat)
            .with_dataset(self._DATASET_NAME)
            .with_captions(im_metadata.captions)
        )

    def len(self) -> int:
        return len(self._captioned_ims)
