import flask
from flask import request

from dream.semsearch import service


_CAPTION = "caption"
_IM_COUNT = "im_count"


def register_semsearch_endpoint(app: flask.Flask, semsearch_svc: service.SemSearchService) -> None:
    @app.route("/semsearch", methods=["GET"])
    def handler():
        payload = request.get_json()

        caption = payload[_CAPTION]
        im_count = payload[_IM_COUNT]

        ids = semsearch_svc.query(caption, n=im_count)

        return ids
