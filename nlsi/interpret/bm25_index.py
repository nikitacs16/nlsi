import random
from typing import Iterable, MutableSequence, Sequence, Tuple

import whoosh.filedb.filestore
import whoosh.index
from whoosh.fields import STORED, TEXT, SchemaClass
from whoosh.qparser import OrGroup, QueryParser

from nlsi.data.datum import NLSIDatum
from nlsi.llm.retrieval.index import DynamicIndex
from nlsi.llm.retrieval.retriever import DynamicIndexRetriever

WHOOSH_TIMEOUT_SECONDS = 2.0


class PromptSchema(SchemaClass):
    text = TEXT()
    key = STORED()


class BM25Index(DynamicIndex[int, str]):
    def __init__(self, seed: int = 12345):
        storage = whoosh.filedb.filestore.RamStorage()  # in-memory storage
        self.index = storage.create_index(schema=PromptSchema)
        self._prng = random.Random(seed)

    def add(self, candidates: Iterable[str]) -> None:
        n = self.index.doc_count()
        with self.index.writer() as writer:
            for i, candidate in enumerate(candidates):
                writer.add_document(text=candidate, key=n + i)  # auto-increment key

    def augment_with_random_samples(
        self, hits: Sequence[Tuple[int, float]], top_k: int
    ) -> Sequence[Tuple[int, float]]:
        if len(hits) < top_k:
            # print(f"Could not retrieve {top_k} examples, got only {len(hits)}")
            keys_to_sample = sorted(
                set(range(self.index.doc_count())).difference(set(k for k, _ in hits))
            )
            sampled_keys = self._prng.sample(
                keys_to_sample, k=min(top_k - len(hits), len(keys_to_sample))
            )
            augmented_hits = list(hits) + [(k, 0.0) for k in sampled_keys]
            # print(f"Added samples to make it of size {len(augmented_hits)}")
            items = augmented_hits[:top_k]
        else:
            items = hits[:top_k]
        return items

    def search(self, query: str, top_k: int = 10) -> Sequence[Tuple[int, float]]:
        with self.index.searcher() as searcher:
            query_parser = QueryParser("text", schema=searcher.schema, group=OrGroup)  # type: ignore
            q = query_parser.parse(query)
            hits = searcher.search(q, limit=top_k)
            results = [(hit["key"], hit.score) for hit in hits]
            return self.augment_with_random_samples(results, top_k=top_k)  # type: ignore

    def save(self, path: str) -> None:
        disk_storage = whoosh.filedb.filestore.FileStorage(path)
        whoosh.filedb.filestore.copy_storage(self.index.storage, disk_storage)
        disk_storage.close()

    def load(self, path: str) -> None:
        disk_storage = whoosh.filedb.filestore.FileStorage(path)
        ram_storage = whoosh.filedb.filestore.copy_to_ram(disk_storage)
        disk_storage.close()
        self.index = ram_storage.open_index(schema=PromptSchema)


class BM25Retriever(DynamicIndexRetriever[str, str, NLSIDatum]):
    def __init__(
        self, train_data: Sequence[NLSIDatum], top_k: int = 20, seed: int = 12345
    ):
        self._data = list(train_data)
        self._index = BM25Index(seed=seed)
        self._top_k = top_k
        self.add(train_data)

    @property
    def index(self) -> DynamicIndex[int, str]:
        return self._index

    @property
    def data(self) -> MutableSequence[NLSIDatum]:
        return self._data

    @property
    def top_k(self) -> int:
        return self._top_k

    def encode_query_batch(self, queries: Sequence[str]) -> Sequence[str]:
        return queries  # [q.natural for q in queries]

    def encode_candidate_batch(self, candidates: Sequence[NLSIDatum]) -> Sequence[str]:
        return [c.user_utterance for c in candidates]


class GeneralBM25Retriever(DynamicIndexRetriever[str, str, NLSIDatum]):
    def __init__(
        self, train_data: Sequence[NLSIDatum], top_k: int = 20, seed: int = 12345
    ):
        self._data = list(train_data)
        self._index = BM25Index(seed=seed)
        self._top_k = top_k
        self.add(train_data)

    @property
    def index(self) -> DynamicIndex[int, str]:
        return self._index

    @property
    def data(self) -> MutableSequence[NLSIDatum]:
        return self._data

    @property
    def top_k(self) -> int:
        return self._top_k

    def encode_query_batch(self, queries: Sequence[str]) -> Sequence[str]:
        return queries  # [q.natural for q in queries]

    def encode_candidate_batch(self, candidates: Sequence[NLSIDatum]) -> Sequence[str]:
        return [c for c in candidates]
