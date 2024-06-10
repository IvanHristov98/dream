from typing import Callable

import psycopg_pool
import psycopg

from dream.experiment import fixture
from dream import pg as dreampg
from dream import model


class ImageMetadataStore(fixture.ImageMetadataStore):
    _pool: psycopg_pool.ConnectionPool

    def __init__(self, pool: psycopg_pool.ConnectionPool) -> None:
        super().__init__()

        self._pool = pool

    def atomically(self, cb: Callable[[any], None]) -> None:
        dreampg.with_tx(self._pool, cb)

    def new_iter(self, tx: any) -> fixture.ImageMetadataIterator:
        pg_tx = dreampg.to_tx(tx)

        query = "SELECT id, captions, dataset FROM image_metadata"
        pg_tx.execute(query)

        return ImageMetadataIterator(pg_tx)


class ImageMetadataIterator(fixture.ImageMetadataIterator):
    _tx: psycopg.Cursor
    _curr: model.Image

    def __init__(self, tx: psycopg.Cursor) -> None:
        super().__init__()

        self._tx = tx

    def next(self) -> bool:
        row = self._tx.fetchone()
        if row is None:
            return False

        self._curr = model.Image().with_id(row[0]).with_captions(row[1]).with_dataset(row[2])
        return True

    def curr(self) -> model.Image:
        return self._curr
