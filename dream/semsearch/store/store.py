import json
from typing import Callable

import psycopg_pool

import dream.semsearch.service as semsearchservice
from dream import model


class Store(semsearchservice.Store):
    _pool: psycopg_pool.ConnectionPool

    def __init__(self, pool: psycopg_pool.ConnectionPool) -> None:
        super().__init__()

        self._pool = pool

    def atomically(self, cb: Callable[[any], None]) -> None:
        # conn wraps a single transaction.
        with self._pool.connection() as conn:
            cur = conn.cursor()
            cb(cur)

    def add_image_metadata(self, tx: any, im: model.Image) -> None:
        persisted_captions = json.dumps(im.captions)

        tx.execute(
            "INSERT INTO image_metadata (id, captions, dataset) VALUES(%s, %s, %s);",
            (im.id.id, persisted_captions, im.dataset),
        )
