import dream.semsearch.service as semsearchservice
from dream import model


class ImageStore(semsearchservice.ImageStore):
    def store_matrix(self, _: model.Image) -> None:
        raise NotImplementedError("")

    def load_matrix(self, _: model.Image) -> model.Image:
        raise NotImplementedError("")
