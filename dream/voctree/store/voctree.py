from typing import Optional, Any
import uuid
import json

import psycopg

import dream.voctree.model as vtmodel
import dream.voctree.service as service
import dream.pg as dreampg
import dream.voctree.store.model as storemodel


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, storemodel.PersistedFeature):
            return obj.__dict__

        return json.JSONEncoder.default(self, obj)


class VocabularyTreeStore(service.VocabularyTreeStore):
    _node_table: str
    _train_job_table: str

    def __init__(self, node_table: str, train_job_table: str) -> None:
        super().__init__()

        self._node_table = node_table
        self._train_job_table = train_job_table

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
            f"INSERT INTO {self._train_job_table} (id, node_id) VALUES (%s, %s)",
            (job.id.hex, job.added_node.id.hex),
        )

        return None

    def fetch_train_job(self, tx: any) -> Optional[vtmodel.TrainJob]:
        pg_tx = dreampg.to_tx(tx)

        # Using FOR UPDATE sets a lock on the row so that it can't be accessed by other transactions while
        # locked. SKIP LOCKED allows a transaction to skip waiting for the lock thus enabling a competing
        # consumers pattern.
        pg_tx.execute(f"SELECT id, node_id FROM {self._train_job_table} LIMIT 1 FOR UPDATE SKIP LOCKED;")
        record = pg_tx.fetchone()
        if record is None:
            return None

        event_id = uuid.UUID(record[0])
        node_id = uuid.UUID(record[1])

        node = self._get_node(pg_tx, node_id)
        if node is None:
            # It shouldn't be possible to encounter this scenario.
            raise RuntimeError("missing node for node_added event")

        return vtmodel.NodeAdded(event_id, node)

    def pop_train_job(self, tx: any, job_id: uuid.UUID) -> None:
        return

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
        return None

    def get_node(self, tx: any, id: uuid.UUID) -> Optional[vtmodel.Node]:
        return self._get_node(tx, id)

    def _get_node(self, pg_tx: psycopg.Cursor, id: uuid.UUID) -> Optional[vtmodel.Node]:
        pg_tx.execute(
            f"""SELECT id, tree_id, depth, vec, children, features
            FROM {self._node_table} WHERE id=%s""",
            (id,),
        )
        record = pg_tx.fetchone()
        if record is None:
            return None

        p_node = storemodel.PersistedNode(record[0], record[1], record[2], record[3], record[4], record[5])

        return storemodel.to_node(p_node)

    def is_training_in_progress(self, tx: any) -> bool:
        return False

    def destroy_obsolete_nodes(self, tx: any) -> None:
        return
