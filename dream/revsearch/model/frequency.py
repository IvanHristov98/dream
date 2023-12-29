import uuid
from typing import NamedTuple
import math


class DocumentFrequency(NamedTuple):
    term_id: uuid.UUID
    unique_docs_count: int
    docs_count: int
