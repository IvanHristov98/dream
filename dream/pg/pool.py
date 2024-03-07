import os

import psycopg_pool


def new_pool() -> psycopg_pool.ConnectionPool:
    conn_info = os.getenv("DB_CONN_INFO", "host=localhost port=5432 dbname=dream user=dream password=devpass")
    return psycopg_pool.ConnectionPool(conn_info)
