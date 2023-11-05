import logging
from typing import NamedTuple

import dream.revsearch.service as revsearch


class Config(NamedTuple):
    sample_size: int


class TrainController:
    _vtree: revsearch.VocabularyTree
    _cfg: Config

    def __init__(self, cfg: Config, vtree: revsearch.VocabularyTree) -> None:
        self._vtree = vtree
        self._cfg = cfg

    def run(self) -> None:
        self._vtree.train(self._cfg.sample_size)
        
