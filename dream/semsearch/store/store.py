from typing import Callable

import dream.semsearch.service as semsearchservice
from dream import model


class Store(semsearchservice.Store):
    def atomically(self, _: Callable[[any], None]) -> None:
        raise NotImplementedError("")

    def add_image_metadata(self, tx: any, im: model.Image) -> None:
        raise NotImplementedError("")
