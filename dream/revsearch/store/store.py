from typing import Any, Callable, List
import json
import logging
import uuid

import psycopg_pool
import psycopg
import numpy as np

from dream.revsearch import service
import dream.revsearch.model as rsmodel
from dream import model


class Store(service.Store):
    _pool: psycopg_pool.ConnectionPool

    def __init__(self, pool: psycopg_pool.ConnectionPool) -> None:
        super().__init__()

        self._pool = pool

    def atomically(self, cb: Callable[[service.TxStore], None]) -> None:
        # conn wraps a single transaction.
        with self._pool.connection() as conn:
            cur = conn.cursor()
            tx_store = TxStore(cur)
            cb(tx_store)


# See https://pynative.com/python-serialize-numpy-ndarray-into-json/
# and https://stackoverflow.com/a/48159596
class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, uuid.UUID):
            return obj.hex

        return json.JSONEncoder.default(self, obj)


class TxStore(service.TxStore):
    _cur: psycopg.Cursor

    def __init__(self, cur: psycopg.Cursor) -> None:
        super().__init__()

        self._cur = cur

    def remove_all_nodes(self) -> None:
        self._cur.execute("DELETE FROM node;")

    def store_node(self, node: rsmodel.Node) -> None:
        child_uuids = list(node.children)
        encoded_child_uuids = json.dumps(child_uuids, cls=JSONEncoder)

        encoded_vec = json.dumps(node.vec, cls=JSONEncoder)
        encoded_features = json.dumps(node.features, cls=JSONEncoder)

        self._cur.execute(
            "INSERT INTO node (id, is_root, children, vec, features) VALUES (%s, %s, %s, %s, %s);",
            (node.id.id, node.is_root, encoded_child_uuids, encoded_vec, encoded_features),
        )

    def find_node(self, node_id: rsmodel.NodeID) -> rsmodel.Node:
        raise NotImplementedError("")

    def store_image_metadata(self, im: model.Image) -> None:
        persisted_labels = json.dumps(im.labels)

        self._cur.execute(
            "INSERT INTO image_metadata (id, labels, dataset) VALUES(%s, %s, %s);",
            (im.id.id, persisted_labels, im.dataset),
        )

    def find_image_metadata(self, im_id: model.ImageID) -> model.Image:
        raise NotImplementedError("")

    def load_training_images(self, sample_size: int) -> List[model.Image]:
        """
        load_training_images currently loads the metadata of all images from the db.
        """
        self._cur.execute("SELECT COUNT(*) FROM image_metadata;")
        row = self._cur.fetchone()
        count = row[0]

        if count < sample_size:
            logging.warn(
                f"sample_size ({sample_size}) is greater than training dataset size ({count}), setting to 100%",
            )
            sample_size = count

        if sample_size / count < 1:
            logging.warn(f"sample_size must be >= 1% of the total number of images, setting it to 1%")
            sample_size = count / 100 + 1

        self._cur.execute(
            f"SELECT id, labels, dataset FROM image_metadata TABLESAMPLE BERNOULLI (({sample_size}*100/{count}));",
        )
        ims = []

        for row in self._cur.fetchall():
            labels = row[1]

            im = model.Image().with_id(model.ImageID(row[0])).with_dataset(row[2]).with_labels(labels)
            ims.append(im)

        return ims
