import flask
from flask import request

from dream.semsearch import service


_CAPTION = "caption"
_IM_COUNT = "im_count"
_INCLUDE_UNLABELED = "include_unlabeled"


def register_images_endpoint(app: flask.Flask, semsearch_svc: service.SemSearchService) -> None:
    @app.route("/images", methods=["GET"])
    def handler():
        payload = request.get_json()

        caption = payload[_CAPTION]
        im_count = payload[_IM_COUNT]
        include_unlabeled = payload[_INCLUDE_UNLABELED]

        if include_unlabeled:
            return semsearch_svc.query_all_ims(caption, n=im_count)

        return semsearch_svc.query_labeled_ims(caption, n=im_count)
