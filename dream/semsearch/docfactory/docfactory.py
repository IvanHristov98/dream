from typing import List
import uuid

from sentence_transformers import SentenceTransformer

from dream.semsearch import service
import dream.voctree.api.voctree as vtapi


class DocumentFactory(service.DocumentFactory):
    _transformer: SentenceTransformer

    def __init__(self) -> None:
        super().__init__()

        self._transformer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def from_caption(self, doc_id: uuid.UUID, captions: List[str]) -> vtapi.Document:
        vectors = self._transformer.encode(captions)
        return vtapi.Document(doc_id, vectors)
