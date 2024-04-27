from typing import NamedTuple
import os

import flask

from dream import logging as dreamlogging
from dream import semsearch
from dream.controller import restapi
from dream import controller


class Config(NamedTuple):
    im_store_path: str
    train_cfg: controller.TrainConfig


def main() -> None:
    dreamlogging.configure_logging()
    cfg = _parse_cfg()

    captions_vtree, ims_vtree, semsearch_svc = semsearch.new_svc(cfg.im_store_path)

    train_ctl = controller.TrainController(cfg.train_cfg, captions_vtree, ims_vtree)
    train_ctl.run()

    app = flask.Flask(__name__)

    restapi.register_train_endpoint(app, captions_vtree, ims_vtree)
    restapi.register_images_endpoint(app, semsearch_svc)

    app.run(host="0.0.0.0", port=5000, debug=False)


def _parse_cfg() -> Config:
    im_store_path = os.getenv("IM_STORE_PATH", "")
    train_cfg = controller.parse_train_cfg()

    return Config(im_store_path, train_cfg)


if __name__ == "__main__":
    main()
