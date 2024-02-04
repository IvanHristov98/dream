import uuid
from typing import Dict

import dream.voctree.service as service
import dream.voctree.model as rsmodel


class FrequencyStore(service.FrequencyStore):
    def set_term(self, tx: any, term_id: uuid.UUID, frequencies: Dict[uuid.UUID, int]) -> None:
        return

    def set_doc_counts(self, tx: any, tree_id: uuid.UUID, doc_counts: int) -> None:
        return

    def get_df(self, tx: any, term_id: uuid.UUID) -> rsmodel.DocumentFrequency:
        return None

    def get_doc_counts(self, tx: any, tree_id: uuid.UUID) -> int:
        return 0

    def get_tfs(self, tx: any, term_id: uuid.UUID) -> Dict[uuid.UUID, int]:
        return None
