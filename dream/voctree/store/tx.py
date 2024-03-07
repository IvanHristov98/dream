from typing import Callable

import psycopg_pool

import dream.voctree.service as service
import dream.pg as dreampg


class TxStore(service.TxStore):
    _pool: psycopg_pool.ConnectionPool

    def __init__(self, pool: psycopg_pool.ConnectionPool) -> None:
        super().__init__()

        self._pool = pool

    def in_tx(self, cb: Callable[[any], None]) -> None:
        dreampg.with_tx(self._pool, cb)
