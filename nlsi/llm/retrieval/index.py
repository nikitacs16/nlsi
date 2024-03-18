
from abc import ABC, abstractmethod
from typing import Generic, Iterable, Sequence, Tuple, TypeVar

Key = TypeVar("Key")
Vector = TypeVar("Vector")


class Index(Generic[Key, Vector], ABC):
    """
    Encapsulates any index that can be searched over.
    It can either be a sparse index (e.g. powered by Whoosh), or a dense index (e.g. powered by FAISS).
    """

    @abstractmethod
    def search(self, query: Vector, top_k: int) -> Sequence[Tuple[Key, float]]:
        raise NotImplementedError

    def search_batch(
        self, queries: Sequence[Vector], top_k: int
    ) -> Sequence[Sequence[Tuple[Key, float]]]:
        """
        May be overridden by subclasses to provide a more efficient implementation.
        """
        return [self.search(q, top_k) for q in queries]

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError


class DynamicIndex(Generic[Key, Vector], Index[Key, Vector]):
    """
    Any index that supports dynamic addition to the set of candidates.
    """

    @abstractmethod
    def add(self, candidates: Iterable[Vector]) -> None:
        """This function should be able to handle streams of candidates being added."""
        raise NotImplementedError
