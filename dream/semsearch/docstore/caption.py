from typing import List

import dream.voctree.api as vtapi


class CaptionStore(vtapi.DocStore):
    def get_documents(self, tx: any, sample_size: int) -> List[vtapi.Document]:
        raise NotImplementedError("")

    def get_feature_dim(self) -> int:
        raise NotImplementedError("")
