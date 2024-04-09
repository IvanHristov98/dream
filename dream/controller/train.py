from typing import NamedTuple
import time
import os

from multiprocessing import Process

import dream.voctree.api as vtapi


_TRAIN_SLEEP_DURATION = 30.0
_REPLACE_TREE_SLEEP_DURATION = 180.0


class TrainConfig(NamedTuple):
    vtree_train_proc_count: int
    ims_sample_size: int
    captions_sample_size: int


def parse_train_cfg() -> TrainConfig:
    vtree_train_proc_count = os.getenv("VTREE_TRAIN_PROC_COUNT", "")
    ims_sample_size = os.getenv("IMS_SAMPLE_SIZE", "")
    captions_sample_size = os.getenv("CAPTIONS_SAMPLE_SIZE", "")

    return TrainConfig(vtree_train_proc_count, ims_sample_size, captions_sample_size)


class TrainController:
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
                time.sleep(_TRAIN_SLEEP_DURATION)

        for _ in range(self._cfg.vtree_train_proc_count):
            proc = Process(target=train, args=(vtree,))
            proc.start()

    def _run_tree_replacers(self, vtree: vtapi.VocabularyTree) -> None:
        def replace_tree(vtree: vtapi.VocabularyTree) -> None:
            while True:
                vtree.try_replace_blue_tree()
                time.sleep(_REPLACE_TREE_SLEEP_DURATION)

        proc = Process(target=replace_tree, args=(vtree,))
        proc.start()
