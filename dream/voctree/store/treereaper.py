from dream.voctree import service

import dream.pg as dreampg
import dream.voctree.store.model as storemodel
import dream.voctree.store.voctree as storevoctree


class TreeReaper(service.TreeReaper):
    _tables: storemodel.TableConfig

    def __init__(self, tables: storemodel.TableConfig) -> None:
        super().__init__()

        self._tables = tables

    def reap_deprecated_trees(self, tx: any) -> None:
        pg_tx = dreampg.to_tx(tx)

        if storevoctree.is_training_in_progress(pg_tx, self._tables):
            return

        pg_tx.execute(f"SELECT id FROM {self._tables.tree_table()} ORDER BY created_at ASC;")
        records = pg_tx.fetchall()

        if len(records) != storemodel.MAX_TREE_COUNT:
            return

        # They are always 2 so we're safe.
        tree_id = records[0][0]

        pg_tx.execute(f"DELETE FROM {self._tables.tf_table()} WHERE tree_id=%s;", (tree_id,))
        pg_tx.execute(f"DELETE FROM {self._tables.df_table()} WHERE tree_id=%s;", (tree_id,))
        pg_tx.execute(f"DELETE FROM {self._tables.tree_doc_count_table()} WHERE tree_id=%s;", (tree_id,))

        pg_tx.execute(f"DELETE FROM {self._tables.node_table()} WHERE tree_id=%s;", (tree_id,))
        pg_tx.execute(f"DELETE FROM {self._tables.tree_table()} WHERE id=%s;", (tree_id,))
