from typing import NamedTuple
import time
import os
from threading import Thread

import dream.voctree.api as vtapi


class TrainConfig(NamedTuple):
    vtree_train_proc_count: int


def parse_train_cfg() -> TrainConfig:
    vtree_train_proc_count = int(os.getenv("VTREE_TRAIN_PROC_COUNT", "1"))
    return TrainConfig(vtree_train_proc_count)


class TrainController:
    _TRAIN_SLEEP_DURATION = 30.0
    _REPLACE_TREE_SLEEP_DURATION = 180.0

    _captions_vtree: vtapi.VocabularyTree
    _ims_vtree: vtapi.VocabularyTree
    _cfg: TrainConfig

    def __init__(self, cfg: TrainConfig, captions_vtree: vtapi.VocabularyTree, ims_vtree: vtapi.VocabularyTree) -> None:
        self._cfg = cfg
        self._captions_vtree = captions_vtree
        self._ims_vtree = ims_vtree

    def run(self) -> None:
        self._start_train_workers(self._captions_vtree)
        self._run_tree_replacers(self._captions_vtree)

        self._start_train_workers(self._ims_vtree)
        self._run_tree_replacers(self._ims_vtree)

    def _start_train_workers(self, vtree: vtapi.VocabularyTree) -> None:
        def train(vtree: vtapi.VocabularyTree) -> None:
            while True:
                vtree.try_training()
                time.sleep(self._TRAIN_SLEEP_DURATION)

        thread = Thread(target=train, args=(vtree,))
        thread.start()

    def _run_tree_replacers(self, vtree: vtapi.VocabularyTree) -> None:
        def replace_tree(vtree: vtapi.VocabularyTree) -> None:
            while True:
                vtree.try_replace_blue_tree()
                time.sleep(self._REPLACE_TREE_SLEEP_DURATION)

        thread = Thread(target=replace_tree, args=(vtree,))
        thread.start()
