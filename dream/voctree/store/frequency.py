import uuid
from typing import Dict, Optional

from dream.voctree import service
import dream.voctree.model as vtmodel
import dream.pg as dreampg


class FrequencyStore(service.FrequencyStore):
    _FETCH_SIZE = 100

    _tf_table: str
    _df_table: str
    _tree_doc_count_table: str

    def __init__(self, tf_table: str, df_table: str, tree_doc_count_table: str) -> None:
        self._tf_table = tf_table
        self._df_table = df_table
        self._tree_doc_count_table = tree_doc_count_table

    def set_term(self, tx: any, term_id: uuid.UUID, frequencies: Dict[uuid.UUID, int]) -> None:
        pg_tx = dreampg.to_tx(tx)

        total_tf = 0
        unique_docs_count = 0

        for doc_id, tf in frequencies.items():
            pg_tx.execute(
                f"INSERT INTO {self._tf_table} (term_id, doc_id, frequency) VALUES (%s, %s, %s);",
                (term_id, doc_id, tf),
            )

            total_tf += tf
            unique_docs_count += 1

        pg_tx.execute(
            f"INSERT INTO {self._df_table} (term_id, unique_docs_count, total_tf) VALUES (%s, %s, %s);",
            (term_id, unique_docs_count, total_tf),
        )

    def set_doc_counts(self, tx: any, tree_id: uuid.UUID, docs_count: int) -> None:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(
            f"INSERT INTO {self._tree_doc_count_table} (tree_id, docs_count) VALUES (%s, %s);", (tree_id, docs_count)
        )

    def get_df(self, tx: any, term_id: uuid.UUID) -> Optional[vtmodel.DocumentFrequency]:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(f"SELECT unique_docs_count, total_tf FROM {self._df_table} WHERE term_id = %s;", (term_id,))
        record = pg_tx.fetchone()
        if record is None:
            return None

        return vtmodel.DocumentFrequency(term_id, record[0], record[1])

    def get_doc_counts(self, tx: any, tree_id: uuid.UUID) -> Optional[int]:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(f"SELECT docs_count FROM {self._tree_doc_count_table} WHERE tree_id = %s;", (tree_id,))
        record = pg_tx.fetchone()
        if record is None:
            return None

        return int(record[0])

    def get_tfs(self, tx: any, term_id: uuid.UUID) -> Dict[uuid.UUID, int]:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(f"SELECT doc_id, frequency FROM {self._tf_table} WHERE term_id = %s;", (term_id,))
        tfs: Dict[uuid.UUID, int] = {}

        while True:
            records = pg_tx.fetchmany(size=self._FETCH_SIZE)

            if records is None or len(records) == 0:
                break

            for record in records:
                tfs[record[0]] = int(record[1])

        return tfs
