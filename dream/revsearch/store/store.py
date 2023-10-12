from typing import Callable, List
import json

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


# See https://pynative.com/python-serialize-numpy-ndarray-into-json/.
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


class TxStore(service.TxStore):
    _cur: psycopg.Cursor

    def __init__(self, cur: psycopg.Cursor) -> None:
        super().__init__()

        self._cur = cur

    def remove_all_nodes(self) -> None:
        self._cur.execute("DELETE FROM node;")

    def store_node(self, node: rsmodel.Node) -> None:
        child_uuids = [None] * len(node.children)

        for i in range(len(node.children)):
            child_uuids[i] = node.children[i]

        encoded_child_uuids = json.dumps(child_uuids, cls=NumpyArrayEncoder)
        encoded_vec = json.dumps(node.vec, cls=NumpyArrayEncoder)
        features = json.dumps(node.features, cls=NumpyArrayEncoder)

        self._cur.execute(
            "INSERT INTO node (id, is_root, children, vec, features) VALUES (%s, %s, %s, %s, %s);",
            (node.id.id, node.is_root, encoded_child_uuids, encoded_vec, features),
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

    def load_training_images(self) -> List[model.Image]:
        """
        load_training_images currently loads the metadata of all images from the db.
        Consider loading a subset if loading too many images becomes a problem.
        """
        self._cur.execute("SELECT id, label, dataset FROM image_metadata;")
        ims = []

        for row in self._cur.fetchall():
            labels = json.loads(row[1])

            im = model.Image().with_id(model.ImageID(row[0])).with_dataset(row[2]).with_labels(labels)
            ims.append(im)

        return ims
