"""
Module service wraps the reverse search use cases according to clean architecture.
"""
from typing import List

from dream import model


class VocabularyTree:
    """
    VocabularyTree implements use cases for reverse search of images.
    """

    def find_similar(self, _: model.ImageFeatures) -> List[model.ImageID]:
        """
        find_similar finds images with features similar to the ones of the input image.
        """
        return None

    def find_image(self, _: model.ImageID) -> model.Image:
        """
        find_image finds an image with a given ID.
        """
        return None
