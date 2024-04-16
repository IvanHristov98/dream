import logging

import flask
from flask import request

from dream.voctree.api import VocabularyTree, ErrTrainingInProgress


_CAPTIONS_SAMPLE_SIZE = "captions_sample_size"
_IMS_SAMPLE_SIZE = "ims_sample_size"


def register_train_endpoint(app: flask.Flask, captions_vtree: VocabularyTree, ims_vtree: VocabularyTree) -> None:
    @app.route("/train", methods=["PUT"])
    def train():
        payload = request.get_json()

        # Only coco is supported as of now. Consider adding a payload parameter when introducing
        # new dataset.
        _start_training(captions_vtree, payload[_CAPTIONS_SAMPLE_SIZE])
        _start_training(ims_vtree, payload[_IMS_SAMPLE_SIZE])

        return ""


def _start_training(vtree: VocabularyTree, sample_size: int) -> None:
    try:
        vtree.start_training(sample_size)
    # Idempotence
    except ErrTrainingInProgress as exc:
        logging.info("training is already in progress %s", str(exc))
