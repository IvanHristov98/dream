import dream.revsearch.service as revsearch
from dream import dataset

from tqdm import tqdm


class SeedController:
    _vtree: revsearch.VocabularyTree
    _ds_iter: dataset.DatasetIterator

    def __init__(self, vtree: revsearch.VocabularyTree, ds_iter: dataset.DatasetIterator) -> None:
        self._vtree = vtree
        self._ds_iter = ds_iter

    def run(self) -> None:
        with tqdm(total=self._ds_iter.len()) as pbar:
            while self._ds_iter.next():
                im = self._ds_iter.read()
                self._vtree.seed_image(im)

                pbar.update(1)
