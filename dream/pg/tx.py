from typing import Callable

import psycopg
import psycopg_pool


def with_tx(pool: psycopg_pool.ConnectionPool, cb: Callable[[any], None]) -> None:
    # TODO: Add retry logic.
    # TODO: See https://pypi.org/project/timeoutcontext/.
    # conn wraps a single transaction.
    with pool.connection() as conn:
        cur = conn.cursor()
        cb(cur)


def to_tx(tx: any) -> psycopg.Cursor:
    if not isinstance(tx, psycopg.Cursor):
        raise AssertionError("tx must be a pg tx cursor")

    return tx
