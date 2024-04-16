"""
Module image provides image models to be used across the whole project.
"""
import uuid
from typing import List

import numpy as np


# NoImageID should be treated as a constant.
NoImageID = uuid.UUID(int=0)


class Image:
    """
    Image represents an image with its ID, description and matrix.
    """

    id: uuid.UUID
    captions: List[str]
    mat: np.ndarray
    dataset: str

    def __init__(self) -> None:
        self.captions = []

    def with_id(self, im_id: uuid.UUID) -> "Image":
        self.id = im_id
        return self

    def with_mat(self, mat: np.ndarray) -> "Image":
        self.mat = mat
        return self

    def with_dataset(self, dataset: np.ndarray) -> "Image":
        self.dataset = dataset
        return self

    def with_captions(self, captions: List[str]) -> "Image":
        """
        with_captions removes all current captions on the model and copies the provided ones as the new captions.
        It should be used only for construction.
        """

        self.captions = []

        for caption in captions:
            self.captions.append(caption)

        return self

    def add_caption(self, caption: str) -> None:
        self.captions.append(caption)
