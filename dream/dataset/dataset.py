import abc

from dream import model


class DatasetIterator(abc.ABC):
    def next(self) -> bool:
        raise NotImplementedError("")

    def read(self) -> model.Image:
        raise NotImplementedError("")

    def len(self) -> int:
        raise NotADirectoryError("")
