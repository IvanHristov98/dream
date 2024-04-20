from typing import List
import uuid

from sentence_transformers import SentenceTransformer

import dream.voctree.api as vtapi
import dream.pg as dreampg
from dream import model
from dream.semsearch.docstore.im_metadata import sample_im_metadata


class CaptionStore(vtapi.DocStore):
    _DIM_SIZE = 384
    _DOC_STORE = "captions"

    _transformer: SentenceTransformer

    def __init__(self) -> None:
        super().__init__()

        self._transformer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def sample_next_documents(self, tx: any, sample_size: int, tree_id: uuid.UUID) -> List[vtapi.Document]:
        pg_tx = dreampg.to_tx(tx)

        ims: List[model.Image] = sample_im_metadata(pg_tx, sample_size, tree_id, self._DOC_STORE)
        docs: List[vtapi.Document] = []

        for im in ims:
            vectors = self._transformer.encode(im.captions)
            doc = vtapi.Document(im.id, vectors)

            docs.append(doc)

        return docs

    def get_feature_dim(self) -> int:
        return self._DIM_SIZE
