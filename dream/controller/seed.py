from tqdm import tqdm

from dream.semsearch import service
from dream import dataset


# TODO: Consider moving to a separate endpoint as a part of the service.
# Could be done at a later stage.
class SeedController:
    _semsearch_svc: service.SemSearchService
    _ds_iter: dataset.DatasetIterator

    def __init__(self, semsearch_svc: service.SemSearchService, ds_iter: dataset.DatasetIterator) -> None:
        self._semsearch_svc = semsearch_svc
        self._ds_iter = ds_iter

    def run(self) -> None:
        with tqdm(total=self._ds_iter.len()) as pbar:
            while self._ds_iter.next():
                im = self._ds_iter.read()
                self._semsearch_svc.seed_image(im)

                pbar.update(1)
