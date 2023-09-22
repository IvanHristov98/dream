from typing import Callable
import json

import psycopg_pool
import psycopg
import numpy as np

import dream.revsearch.service as service
import dream.revsearch.model as rsmodel
import dream.model as model


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

    def store_node(self, node: rsmodel.Node) -> None:
        child_uuids = [None] * len(node.children)

        for i in range(len(node.children)):
            child_uuids[i] = node.children[i]

        encoded_child_uuids = json.dumps(child_uuids, cls=NumpyArrayEncoder)
        encoded_vec = json.dumps(node.vec, cls=NumpyArrayEncoder)
        features = json.dumps(node.features, cls=NumpyArrayEncoder)

        self._cur.execute(
            "INSERT INTO node (id, children, vec, features) VALUES (%s, %s, %s, %s);",
            (node.id.id, encoded_child_uuids, encoded_vec, features),
        )

    def find_node(self, node_id: rsmodel.NodeID) -> rsmodel.Node:
        raise NotImplementedError("")

    def store_image_metadata(self, im: model.Image) -> None:
        self._cur.execute(
            "INSERT INTO image_metadata (id, label) VALUES(%s, %s);",
            (im.id.id, im.label),
        )

    def find_image_metadata(self, im_id: model.ImageID) -> model.Image:
        raise NotImplementedError("")
