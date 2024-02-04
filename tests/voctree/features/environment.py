from typing import Any

from behave import fixture
from behave import use_fixture

import dream.pg as dreampg


@fixture
def conn_pool(context, *args, **kwargs):
    try:
        context.conn_pool = dreampg.new_pool()
        yield context.conn_pool
    finally:
        context.conn_pool.close()


def migrate_up_db(context):
    def _cb(tx: Any) -> None:
        cur = dreampg.to_tx(tx)
        cur.execute(
            f"""CREATE TABLE IF NOT EXISTS test_node (
            id       UUID  PRIMARY KEY,
            tree_id  UUID NOT NULL,
            depth    INT NOT NULL,
            vec      JSONB NOT NULL,
            children JSONB NOT NULL,
            features JSONB NOT NULL);"""
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS test_node_tree_id_depth_idx ON test_node (tree_id, depth);")
        cur.execute(
            f"""CREATE TABLE IF NOT EXISTS test_train_job (
            id         UUID PRIMARY KEY,
            node_id    UUID NOT NULL,

            CONSTRAINT fk_node_id FOREIGN KEY (node_id) REFERENCES test_node(id));"""
        )

    dreampg.with_tx(context.conn_pool, _cb)


def migrate_down_db(context):
    def _cb(tx: Any) -> None:
        cur = dreampg.to_tx(tx)
        cur.execute("DROP TABLE IF EXISTS test_train_job;")
        cur.execute("DROP TABLE IF EXISTS test_node;")

    dreampg.with_tx(context.conn_pool, _cb)


def cleanup_db(context):
    # TODO: Come up with a solution where tests could be run in parallel.
    def _cb(tx: Any) -> None:
        cur = dreampg.to_tx(tx)
        cur.execute("DELETE FROM test_train_job;")
        cur.execute("DELETE FROM test_node;")

    dreampg.with_tx(context.conn_pool, _cb)


def before_tag(context, tag):
    if tag == "fixture.conn.pool":
        use_fixture(conn_pool, context)
    elif tag == "db.schema":
        migrate_up_db(context)


def after_tag(context, tag):
    if tag == "db.schema":
        migrate_down_db(context)
    elif tag == "db.cleanup":
        cleanup_db(context)
