import json
import uuid
from typing import Callable

import psycopg_pool

import dream.semsearch.service as semsearchservice
from dream import model
import dream.pg as dreampg


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
            (im.id, persisted_captions, im.dataset),
        )

    def get_image_metadata(self, tx: any, im_id: uuid.UUID) -> model.Image:
        pg_tx = dreampg.to_tx(tx)

        query = "SELECT captions, dataset FROM image_metadata im WHERE im.id = %s"
        pg_tx.execute(query, (im_id,))

        row = pg_tx.fetchone()
        if row is None:
            return None

        return model.Image().with_id(im_id).with_dataset(row[1]).with_captions(row[0])
