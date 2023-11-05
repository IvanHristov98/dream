"""
Module image provides image models to be used across the whole project.
"""
import uuid
from typing import NamedTuple, List

import numpy as np


class ImageID(NamedTuple):
    """
    ImageID represents an ImageID. It could be used for comparison.
    """

    id: uuid.UUID

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "ImageID"):
        return self.id == other.id

    def __str__(self) -> str:
        return str(self.id)


# NoImageID should be treated as a constant.
NoImageID = ImageID(id=uuid.UUID(int=0))


def new_image_id() -> ImageID:
    """
    new_image_id returns a random ImageID.
    """
    return ImageID(uuid.uuid4())


class Image:
    """
    Image represents an image with its ID, description and matrix.
    """

    id: ImageID
    labels: List[str]
    mat: np.ndarray
    dataset: str

    def __init__(self) -> None:
        self.labels = []

    def with_id(self, id: ImageID) -> "Image":
        self.id = id
        return self

    def with_mat(self, mat: np.ndarray) -> "Image":
        self.mat = mat
        return self

    def with_dataset(self, dataset: np.ndarray) -> "Image":
        self.dataset = dataset
        return self

    def with_labels(self, labels: List[str]) -> "Image":
        """
        with_labels removes all current labels on the model and copies the provided ones as the new labels.
        It should be used only for construction.
        """

        self.labels = []

        for label in labels:
            self.labels.append(label)

        return self

    def add_label(self, label: str) -> None:
        self.labels.append(label)
