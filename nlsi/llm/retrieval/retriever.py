import asyncio
import dataclasses
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Awaitable,
    Generic,
    Iterable,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
)

import cachetools
from more_itertools import chunked

from nlsi.llm.retrieval.index import DynamicIndex, Index, Vector

Query = TypeVar("Query", contravariant=True)
Candidate = TypeVar("Candidate")


DEFAULT_BATCH_SIZE = 32


class DataFilter(Generic[Query, Candidate], ABC):
    """
    Executes some post-processing step after a batch of candidates are retrieved.
    It may include reranking, or pruning candidates that are too long to fit into the prompt, etc.
    """

    @abstractmethod
    async def __call__(
        self, candidates: Sequence[Candidate], query: Query
    ) -> Sequence[Candidate]:
        raise NotImplementedError


class IdentityDataFilter(DataFilter[Query, Candidate]):
    """
    A no-op data filter that does nothing.
    """

    async def __call__(
        self, candidates: Sequence[Candidate], query: Query
    ) -> Sequence[Candidate]:
        return candidates


class DataRetriever(Generic[Query, Candidate], ABC):
    """
    The basic interface for a data retriever for few-shot candidate retrieval.
    """

    async def retrieve(self, query: Query) -> Sequence[Candidate]:
        raise NotImplementedError

    async def retrieve_batch(
        self, queries: Sequence[Query]
    ) -> Sequence[Sequence[Candidate]]:
        """Override this method to provide a more efficient implementation."""
        return await asyncio.gather(*[self.retrieve(q) for q in queries])

    def then(self, df: DataFilter) -> "DataRetriever[Query, Candidate]":
        """
        Returns a new data retriever that applies the given data filter to the results of this data retriever.
        """
        return PipelinedDataRetriever(self, [df])

    def cached(self) -> "DataRetriever[Query, Candidate]":
        """
        Returns a new data retriever that caches the results of this data retriever.
        """
        return CachedDataRetriever(self)

    def __rshift__(self, df: DataFilter) -> "DataRetriever[Query, Candidate]":
        """A symbolic alias for `then`."""
        return self.then(df)


@dataclass
class CachedDataRetriever(Generic[Query, Candidate], DataRetriever[Query, Candidate]):
    """
    A data retriever that caches the results of the underlying data retriever.
    This is useful for when multiple prompts will be sampled for the same datum.
    """

    underlying: DataRetriever[Query, Candidate]
    cache: cachetools.Cache = dataclasses.field(
        default_factory=lambda: cachetools.LRUCache(100)
    )

    def _underlying_task(self, query: Query) -> Awaitable[Sequence[Candidate]]:
        return asyncio.create_task(self.underlying.retrieve(query))

    async def retrieve(self, query: Query) -> Sequence[Candidate]:
        if query not in self.cache:
            self.cache[query] = self._underlying_task(query)
        return await self.cache[query]


class SampledDataRetriever(Generic[Query, Candidate], DataRetriever[Query, Candidate]):
    """
    A data retriever that samples from a mixture of the underlying retriever and
    random candidates.  For now this is very simple, taking a random proportion of the
    ranked candidates and filling the rest with random, but there are likely
    better strategies.  The intended use case is models that ensemble over prompts.
    """

    def __init__(
        self,
        underlying: DataRetriever[Query, Candidate],
        data: Sequence[Candidate],
        k: int,
        random_seed: int,
    ):
        self._underlying = underlying
        self._data = data
        self._k = k
        self._random_seed = random_seed

    async def retrieve(self, query: Query) -> Sequence[Candidate]:
        rand = random.Random(self._random_seed)
        candidates = await self._underlying.retrieve(query)
        n = int(len(candidates) * rand.random())
        sample = rand.sample(self._data, self._k - n)
        return list(candidates[:n]) + list(sample)


class PipelinedDataRetriever(
    Generic[Query, Candidate], DataRetriever[Query, Candidate]
):
    def __init__(
        self,
        underlying: DataRetriever[Query, Candidate],
        filters: Sequence[DataFilter[Query, Candidate]],
    ):
        self.underlying = underlying
        self.filters = filters

    async def _filter(
        self, data: Sequence[Candidate], query: Query
    ) -> Sequence[Candidate]:
        filtered = data
        for f in self.filters:
            filtered = await f(filtered, query)
        return filtered

    async def retrieve_batch(
        self, queries: Sequence[Query]
    ) -> Sequence[Sequence[Candidate]]:
        retrieved = await self.underlying.retrieve_batch(queries)
        filtered = await asyncio.gather(
            *[self._filter(r, q) for r, q in zip(retrieved, queries)]
        )
        return filtered

    async def retrieve(self, query: Query) -> Sequence[Candidate]:
        retrieved = await self.underlying.retrieve(query)
        filtered = await self._filter(retrieved, query)
        return filtered

    def then(self, df: DataFilter) -> "DataRetriever[Query, Candidate]":
        return PipelinedDataRetriever(self.underlying, list(self.filters) + [df])


# NOTE: Type variable "Vector" should be an existential type inside the class,
# NOTE: but this fails pyright typechecking.
# NOTE: So have to make "Vector" a universal type instead.
class IndexRetriever(
    DataRetriever[Query, Candidate], Generic[Vector, Query, Candidate]
):
    @property
    @abstractmethod
    def index(self) -> Index[int, Vector]:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> Sequence[Candidate]:
        raise NotImplementedError

    @property
    @abstractmethod
    def top_k(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode_query_batch(self, queries: Sequence[Query]) -> Sequence[Vector]:
        raise NotImplementedError

    def encode_query(self, query: Query) -> Vector:
        return self.encode_query_batch([query])[0]

    @abstractmethod
    def encode_candidate_batch(
        self, candidates: Sequence[Candidate]
    ) -> Sequence[Vector]:
        raise NotImplementedError

    def encode_candidate(self, candidate: Candidate) -> Vector:
        return self.encode_candidate_batch([candidate])[0]

    async def retrieve(
        self, query: Query, top_k: Optional[int] = None
    ) -> Sequence[Candidate]:
        if top_k is None:
            top_k = self.top_k
        result = self.index.search(self.encode_query(query), top_k)  # self.top_k)
        return [self.data[k] for k, _ in result]

    async def retrieve_batch(
        self, queries: Sequence[Query]
    ) -> Sequence[Sequence[Candidate]]:
        results = self.index.search_batch(self.encode_query_batch(queries), self.top_k)
        return [[self.data[k] for k, _ in result] for result in results]


class DynamicIndexRetriever(
    Generic[Vector, Query, Candidate], IndexRetriever[Vector, Query, Candidate], ABC
):
    """
    Abstract class for any retriever that supports dynamic addition of candidates.
    """

    @property
    @abstractmethod
    def index(self) -> DynamicIndex[int, Vector]:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> MutableSequence[Candidate]:
        raise NotImplementedError

    def add(self, candidates: Iterable[Candidate]) -> None:
        for batch in chunked(candidates, DEFAULT_BATCH_SIZE):
            self.index.add(self.encode_candidate_batch(batch))
            self.data.extend(batch)
