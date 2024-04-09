from typing import Optional, Any
import uuid
import json

import psycopg

import dream.voctree.model as vtmodel
from dream.voctree import service
import dream.pg as dreampg
import dream.voctree.store.model as storemodel
import dream.voctree.api as vtapi


_MAX_TREE_COUNT = 2


class ErrTreeCountOverflow(Exception):
    """
    ErrTreeCountOverflow is thrown when more than the maximum number of trees are attempted to be added.
    """


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, storemodel.PersistedFeature):
            return o.__dict__

        return json.JSONEncoder.default(self, o)


class VocabularyTreeStore(service.VocabularyTreeStore):
    _tree_table: str
    _node_table: str
    _train_job_table: str

    def __init__(self, tree_table: str, node_table: str, train_job_table: str) -> None:
        super().__init__()

        self._tree_table = tree_table
        self._node_table = node_table
        self._train_job_table = train_job_table

    def add_tree(self, tx: any, tree_id: uuid.UUID) -> None:
        pg_tx = dreampg.to_tx(tx)

        if self._is_training_in_progress(pg_tx):
            raise vtapi.ErrTrainingInProgress(f"inserting {tree_id} while training is in progress")

        if self._get_tree_count(pg_tx) == _MAX_TREE_COUNT:
            raise ErrTreeCountOverflow(f"inserting more than {_MAX_TREE_COUNT} trees")

        pg_tx.execute(f"INSERT INTO {self._tree_table} (id) VALUES (%s)", (tree_id,))

    def _is_training_in_progress(self, pg_tx: psycopg.Cursor) -> bool:
        pg_tx.execute(f"SELECT COUNT(1) FROM {self._train_job_table};")
        record = pg_tx.fetchone()
        if record is None:
            raise RuntimeError("expected to receive at least one row on check for training in progress")

        return record[0] > 0

    def _get_tree_count(self, pg_tx: psycopg.Cursor) -> int:
        pg_tx.execute(f"SELECT COUNT(1) FROM {self._tree_table};")
        record = pg_tx.fetchone()
        if record is None:
            raise RuntimeError("expected to receive at least one row when getting tree count")

        return record[0]

    def add_node(self, tx: any, node: vtmodel.Node) -> None:
        pg_tx = dreampg.to_tx(tx)

        p_node = storemodel.to_persisted_node(node)

        encoded_vec = json.dumps(p_node.vec)
        encoded_children = json.dumps(p_node.children)
        encoded_features = json.dumps(p_node.features, cls=JSONEncoder)

        pg_tx.execute(
            f"""INSERT INTO {self._node_table} (id, tree_id, depth, vec, children, features)
            VALUES (%s, %s, %s, %s, %s, %s);""",
            (p_node.id, p_node.tree_id, p_node.depth, encoded_vec, encoded_children, encoded_features),
        )

    def push_train_job(self, tx: any, job: vtmodel.TrainJob) -> None:
        pg_tx = dreampg.to_tx(tx)
        pg_tx.execute(
            f"INSERT INTO {self._train_job_table} (id, node_id) VALUES (%s, %s);",
            (job.id, job.added_node.id),
        )

    def fetch_train_job(self, tx: any) -> Optional[vtmodel.TrainJob]:
        pg_tx = dreampg.to_tx(tx)

        # Using FOR UPDATE sets a lock on the row so that it can't be accessed by other transactions while
        # locked. SKIP LOCKED allows a transaction to skip waiting for the lock thus enabling a competing
        # consumers pattern.
        pg_tx.execute(f"SELECT id, node_id FROM {self._train_job_table} LIMIT 1 FOR UPDATE SKIP LOCKED;")
        record = pg_tx.fetchone()
        if record is None:
            return None

        node = self._get_node(pg_tx, record[1])
        if node is None:
            # It shouldn't be possible to encounter this scenario.
            raise RuntimeError("missing node for node_added event")

        return vtmodel.TrainJob(record[0], node)

    def pop_train_job(self, tx: any, job_id: uuid.UUID) -> None:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(f"DELETE FROM {self._train_job_table} WHERE id=%s;", (job_id,))

    def update_node(self, tx: any, node: vtmodel.Node) -> None:
        pg_tx = dreampg.to_tx(tx)

        p_node = storemodel.to_persisted_node(node)

        encoded_vec = json.dumps(p_node.vec)
        encoded_children = json.dumps(p_node.children)
        encoded_features = json.dumps(p_node.features, cls=JSONEncoder)

        pg_tx.execute(
            f"""UPDATE {self._node_table} SET tree_id=%s, depth=%s, vec=%s, children=%s, features=%s WHERE id=%s;""",
            (p_node.tree_id, p_node.depth, encoded_vec, encoded_children, encoded_features, p_node.id),
        )

    def get_root(self, tx: any) -> Optional[vtmodel.Node]:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(
            f"""SELECT n.id, n.tree_id, n.depth, n.vec, n.children, n.features 
            FROM {self._node_table} AS n
            WHERE n.depth=0 AND n.tree_id NOT IN (
                SELECT n2.tree_id 
                FROM {self._node_table} AS n2 JOIN {self._train_job_table} tj ON n2.id = tj.node_id
                LIMIT 1
            )
            LIMIT 1;"""
        )
        return self._read_node(pg_tx)

    def get_node(self, tx: any, node_id: uuid.UUID) -> Optional[vtmodel.Node]:
        return self._get_node(tx, node_id)

    def _get_node(self, pg_tx: psycopg.Cursor, node_id: uuid.UUID) -> Optional[vtmodel.Node]:
        pg_tx.execute(
            f"""SELECT id, tree_id, depth, vec, children, features
            FROM {self._node_table} WHERE id=%s""",
            (node_id,),
        )
        return self._read_node(pg_tx)

    def _read_node(self, pg_tx: psycopg.Cursor) -> Optional[vtmodel.Node]:
        record = pg_tx.fetchone()
        if record is None:
            return None

        p_node = storemodel.PersistedNode(record[0], record[1], record[2], record[3], record[4], record[5])

        return storemodel.to_node(p_node)

    def cleanup(self, tx: any) -> None:
        pg_tx = dreampg.to_tx(tx)

        if self._is_training_in_progress(pg_tx):
            return

        pg_tx.execute(f"SELECT id FROM {self._tree_table};")
        records = pg_tx.fetchall()

        if len(records) != _MAX_TREE_COUNT:
            return

        # They are always 2 so we're safe.
        tree_id = records[0][0]
        pg_tx.execute(f"DELETE FROM {self._node_table} WHERE tree_id=%s;", (tree_id,))
        pg_tx.execute(f"DELETE FROM {self._tree_table} WHERE id=%s;", (tree_id,))
