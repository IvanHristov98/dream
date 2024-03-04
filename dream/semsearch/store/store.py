import dream.semsearch.service as semsearchservice


class Store(semsearchservice.Store):
    def add_image(self, tx: any) -> None:
        raise NotImplementedError("add_document is not implemented")
