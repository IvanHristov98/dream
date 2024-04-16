from typing import Any

from behave import fixture
from behave import use_fixture
import psycopg

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
        _create_tree_table(cur)
        _create_node_table(cur)
        _create_train_job_table(cur)

        _create_tf_table(cur)
        _create_df_table(cur)
        _create_tree_doc_count_table(cur)

    dreampg.with_tx(context.conn_pool, _cb)


def _create_tree_table(tx: psycopg.Cursor) -> None:
    tx.execute(
        """CREATE TABLE IF NOT EXISTS test_tree (
        id UUID PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());"""
    )


def _create_node_table(tx: psycopg.Cursor) -> None:
    tx.execute(
        """CREATE TABLE IF NOT EXISTS test_node (
        id         UUID  PRIMARY KEY,
        tree_id    UUID NOT NULL,
        depth      INT NOT NULL,
        vec        JSONB NOT NULL,
        children   JSONB NOT NULL,
        features   JSONB NOT NULL,
    
        CONSTRAINT test_node_tree_id_fk FOREIGN KEY (tree_id) REFERENCES test_tree(id));"""
    )
    # Index is used when fetching root.
    tx.execute(f"CREATE INDEX IF NOT EXISTS test_node_depth_tree_id_idx ON test_node (depth, tree_id);")


def _create_train_job_table(tx: psycopg.Cursor) -> None:
    tx.execute(
        """CREATE TABLE IF NOT EXISTS test_train_job (
        id         UUID PRIMARY KEY,
        node_id    UUID NOT NULL,

        CONSTRAINT test_train_job_node_id_fk FOREIGN KEY (node_id) REFERENCES test_node(id));"""
    )


def _create_tf_table(tx: psycopg.Cursor) -> None:
    tx.execute(
        """CREATE TABLE IF NOT EXISTS test_tf (
        term_id   UUID NOT NULL,
        doc_id    UUID NOT NULL,
        frequency INT NOT NULL,
        tree_id   UUID NOT NULL,

        CONSTRAINT test_tf_tree_id_fk FOREIGN KEY (tree_id) REFERENCES test_tree(id),
        PRIMARY KEY (term_id, doc_id));"""
    )


def _create_df_table(tx: psycopg.Cursor) -> None:
    tx.execute(
        """CREATE TABLE IF NOT EXISTS test_df (
        term_id           UUID PRIMARY KEY,
	    unique_docs_count INT NOT NULL,
	    total_tf          INT NOT NULL,
        tree_id           UUID NOT NULL,

	    CONSTRAINT test_df_tree_id_fk FOREIGN KEY (tree_id) REFERENCES test_tree(id)
        );"""
    )


def _create_tree_doc_count_table(tx: psycopg.Cursor) -> None:
    tx.execute(
        """CREATE TABLE IF NOT EXISTS test_tree_doc_count (
        tree_id UUID PRIMARY KEY,
        docs_count INT NOT NULL,
        
        CONSTRAINT fk_tree_id FOREIGN KEY (tree_id) REFERENCES test_tree(id));"""
    )


def migrate_down_db(context):
    def _cb(tx: Any) -> None:
        cur = dreampg.to_tx(tx)
        cur.execute("DROP TABLE IF EXISTS test_tree_doc_count;")
        cur.execute("DROP TABLE IF EXISTS test_df;")
        cur.execute("DROP TABLE IF EXISTS test_tf;")

        cur.execute("DROP TABLE IF EXISTS test_train_job;")
        cur.execute("DROP TABLE IF EXISTS test_node;")
        cur.execute("DROP TABLE IF EXISTS test_tree;")

    dreampg.with_tx(context.conn_pool, _cb)


def cleanup_db(context):
    # TODO: Come up with a solution where tests could be run in parallel.
    def _cb(tx: Any) -> None:
        cur = dreampg.to_tx(tx)
        cur.execute("DELETE FROM test_tree_doc_count;")
        cur.execute("DELETE FROM test_df;")
        cur.execute("DELETE FROM test_tf;")

        cur.execute("DELETE FROM test_train_job;")
        cur.execute("DELETE FROM test_node;")
        cur.execute("DELETE FROM test_tree;")

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
