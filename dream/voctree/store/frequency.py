import uuid
from typing import Dict, Optional

from dream.voctree import service
import dream.voctree.model as vtmodel
import dream.pg as dreampg
from dream.voctree.store import model as storemodel


class FrequencyStore(service.FrequencyStore):
    _FETCH_SIZE = 100

    _tables: storemodel.TableConfig

    def __init__(self, tables: storemodel.TableConfig) -> None:
        self._tables = tables

    def set_term(self, tx: any, term_id: uuid.UUID, frequencies: Dict[uuid.UUID, int], tree_id: uuid.UUID) -> None:
        pg_tx = dreampg.to_tx(tx)

        total_tf = 0
        unique_docs_count = 0

        for doc_id, tf in frequencies.items():
            pg_tx.execute(
                f"INSERT INTO {self._tables.tf_table()} (term_id, doc_id, frequency, tree_id) VALUES (%s, %s, %s, %s);",
                (term_id, doc_id, tf, tree_id),
            )

            total_tf += tf
            unique_docs_count += 1

        pg_tx.execute(
            f"""INSERT INTO {self._tables.df_table()} (term_id, unique_docs_count, total_tf, tree_id) 
                VALUES (%s, %s, %s, %s);""",
            (term_id, unique_docs_count, total_tf, tree_id),
        )

    def set_doc_counts(self, tx: any, tree_id: uuid.UUID, docs_count: int) -> None:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(
            f"INSERT INTO {self._tables.tree_doc_count_table()} (tree_id, docs_count) VALUES (%s, %s);",
            (tree_id, docs_count),
        )

    def get_df(self, tx: any, term_id: uuid.UUID) -> Optional[vtmodel.DocumentFrequency]:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(
            f"SELECT unique_docs_count, total_tf FROM {self._tables.df_table()} WHERE term_id = %s;",
            (term_id,),
        )
        record = pg_tx.fetchone()
        if record is None:
            return None

        return vtmodel.DocumentFrequency(term_id, record[0], record[1])

    def get_doc_counts(self, tx: any, tree_id: uuid.UUID) -> Optional[int]:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(f"SELECT docs_count FROM {self._tables.tree_doc_count_table()} WHERE tree_id = %s;", (tree_id,))
        record = pg_tx.fetchone()
        if record is None:
            return None

        return int(record[0])

    def get_tfs(self, tx: any, term_id: uuid.UUID) -> Dict[uuid.UUID, int]:
        pg_tx = dreampg.to_tx(tx)

        pg_tx.execute(f"SELECT doc_id, frequency FROM {self._tables.tf_table()} WHERE term_id = %s;", (term_id,))
        tfs: Dict[uuid.UUID, int] = {}

        while True:
            records = pg_tx.fetchmany(size=self._FETCH_SIZE)

            if records is None or len(records) == 0:
                break

            for record in records:
                tfs[record[0]] = int(record[1])

        return tfs
