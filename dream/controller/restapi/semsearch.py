import uuid

import flask
from flask import request

from dream.semsearch import service


_CAPTION_ARG = "caption"
_IM_COUNT_ARG = "im_count"
_INCLUDE_UNLABELED_ARG = "include_unlabeled"
_IM_ID_FIELD = "im_id"
_DATASET_FIELD = "dataset"
_CAPTIONS_FIELD = "captions"


def register_images_endpoint(app: flask.Flask, semsearch_svc: service.SemSearchService) -> None:
    @app.route("/images", methods=["GET"])
    def images():
        caption = request.args.get(_CAPTION_ARG, type=str)
        im_count = request.args.get(_IM_COUNT_ARG, type=int)

        include_unlabeled = False

        if request.args.get(_INCLUDE_UNLABELED_ARG) is not None:
            include_unlabeled = True

        if include_unlabeled:
            resp = flask.jsonify(semsearch_svc.query_all_ims(caption, n=im_count))
        else:
            resp = flask.jsonify(semsearch_svc.query_labeled_ims(caption, n=im_count))

        # TODO: Fix when going to production.
        resp.headers.add("Access-Control-Allow-Origin", "*")

        return resp


def register_image_metadata_endpoint(app: flask.Flask, semsearch_svc: service.SemSearchService) -> None:
    @app.route("/images/<im_id>", methods=["GET"])
    def image_metadata(im_id):
        im_id = uuid.UUID(im_id)
        im = semsearch_svc.get_im_metadata(im_id)

        payload = {
            _IM_ID_FIELD: im.id,
            _DATASET_FIELD: im.dataset,
            _CAPTIONS_FIELD: im.captions,
        }

        resp = flask.jsonify(payload)
        # TODO: Fix when going to production.
        resp.headers.add("Access-Control-Allow-Origin", "*")

        return resp


def register_image_endpoint(app: flask.Flask, semsearch_svc: service.SemSearchService) -> None:
    @app.route("/mat/<im_id>")
    def image(im_id):
        im_path = semsearch_svc.get_im_path(im_id)

        return flask.send_from_directory(im_path.parent, im_path.name)
