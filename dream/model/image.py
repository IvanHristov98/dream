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


class Image(NamedTuple):
    """
    Image represents an image with its ID, description and matrix.
    """

    id: ImageID
    label: str
    mat: np.ndarray
