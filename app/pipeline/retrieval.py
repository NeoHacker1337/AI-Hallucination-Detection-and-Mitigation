"""Stage 2: evidence retrieval with sentence-transformers embeddings and FAISS.

Passages are stored in SQLite (`chunks` table); each collection has its own FAISS index file whose
vector ids are the chunk ids. If an index file is missing or was built with another embedding
model it is rebuilt from SQLite.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import delete, func, select

from app.config import get_settings
from app.db import session_scope
from app.db.models import Chunk, Document

if TYPE_CHECKING:
    import faiss  # noqa: F401


@dataclass
class Passage:
    """A retrieved passage with its source id and similarity score."""

    chunk_id: int
    source_id: str
    title: str
    text: str
    score: float
    origin: str = "kb"  # "kb" (local knowledge base) or "web"
    source: str | None = None  # document source / URL as recorded when it was added
    last_modified: str | None = None  # ISO date of the content, when the source reports one
    added_at: str | None = None  # ISO time the document was added to the knowledge base

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentInput:
    """A document to add to the knowledge base."""

    title: str
    text: str
    source: str | None = None
    content_date: str | None = None  # ISO date of the content (used for freshness checks)


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split text into overlapping word windows, preferring to break at sentence ends."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    words = [w for w in words if w]
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        if end < len(words):
            # move the end back to the last sentence boundary inside the last third of the window
            for j in range(end, start + (2 * chunk_words) // 3, -1):
                if words[j - 1].endswith((".", "!", "?")):
                    end = j
                    break
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _safe_name(collection: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", collection)


class KnowledgeBase:
    """Vector store over document chunks, one FAISS index per collection."""

    def __init__(self, index_dir: Path | None = None, embedding_model: str | None = None) -> None:
        settings = get_settings()
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.chunk_words = settings.chunk_words
        self.overlap_words = settings.chunk_overlap_words
        self._embedder: Any = None
        self._indexes: dict[str, Any] = {}
        self._lock = threading.RLock()

    # -- embeddings ------------------------------------------------------------------------
    def _get_embedder(self) -> Any:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is not installed; see requirements.txt") from exc
            from app.pipeline.nli import resolve_device

            try:
                self._embedder = SentenceTransformer(self.embedding_model_name, device=resolve_device())
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Could not load embedding model '{self.embedding_model_name}': {exc}. "
                    "Check HG_EMBEDDING_MODEL and your internet connection / HF cache."
                ) from exc
        return self._embedder

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised float32 embeddings (inner product = cosine similarity)."""
        vectors = self._get_embedder().encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

    # -- index management -----------------------------------------------------------------
    def _paths(self, collection: str) -> tuple[Path, Path]:
        base = self.index_dir / _safe_name(collection)
        return base.with_suffix(".faiss"), base.with_suffix(".meta.json")

    def _new_index(self, dim: int) -> Any:
        import faiss

        return faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def _save(self, collection: str, index: Any) -> None:
        import faiss

        index_path, meta_path = self._paths(collection)
        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps({"embedding_model": self.embedding_model_name, "ntotal": index.ntotal}))

    def _rebuild(self, collection: str) -> Any:
        with session_scope() as session:
            rows = session.execute(
                select(Chunk.id, Chunk.text).where(Chunk.collection == collection).order_by(Chunk.id)
            ).all()
        if not rows:
            return None
        ids = np.array([r[0] for r in rows], dtype="int64")
        vectors = self.embed([r[1] for r in rows])
        index = self._new_index(vectors.shape[1])
        index.add_with_ids(vectors, ids)
        self._save(collection, index)
        return index

    def _load(self, collection: str) -> Any:
        with self._lock:
            if collection in self._indexes:
                return self._indexes[collection]
            import faiss

            index_path, meta_path = self._paths(collection)
            index = None
            if index_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("embedding_model") == self.embedding_model_name:
                    index = faiss.read_index(str(index_path))
                    if index.ntotal != self.count(collection):
                        index = None
            if index is None:
                index = self._rebuild(collection)
            if index is not None:
                self._indexes[collection] = index
            return index

    # -- public API --------------------------------------------------------------------------
    def count(self, collection: str) -> int:
        """Number of chunks stored in a collection."""
        with session_scope() as session:
            return int(
                session.execute(select(func.count(Chunk.id)).where(Chunk.collection == collection)).scalar() or 0
            )

    def list_collections(self) -> dict[str, int]:
        """Collection name -> number of chunks."""
        with session_scope() as session:
            rows = session.execute(select(Chunk.collection, func.count(Chunk.id)).group_by(Chunk.collection)).all()
        return {r[0]: int(r[1]) for r in rows}

    def add_documents(self, collection: str, documents: list[DocumentInput]) -> list[int]:
        """Chunk, store and index documents. Returns the new document ids."""
        with self._lock:
            new_chunks: list[tuple[int, str]] = []
            doc_ids: list[int] = []
            with session_scope() as session:
                for doc in documents:
                    pieces = chunk_text(doc.text, self.chunk_words, self.overlap_words)
                    if not pieces:
                        continue
                    row = Document(collection=collection, title=doc.title[:500], source=doc.source,
                                   content_date=doc.content_date, n_chunks=len(pieces))
                    session.add(row)
                    session.flush()
                    doc_ids.append(row.id)
                    for i, piece in enumerate(pieces):
                        chunk = Chunk(document_id=row.id, collection=collection, chunk_index=i, text=piece)
                        session.add(chunk)
                        session.flush()
                        new_chunks.append((chunk.id, piece))
            if not new_chunks:
                return doc_ids
            index = self._load(collection)
            if index is None:  # _load rebuilt from DB, which already contains the new chunks
                index = self._rebuild(collection)
                self._indexes[collection] = index
                return doc_ids
            if index.ntotal == self.count(collection):  # rebuild already included them
                return doc_ids
            vectors = self.embed([t for _, t in new_chunks])
            index.add_with_ids(vectors, np.array([i for i, _ in new_chunks], dtype="int64"))
            self._save(collection, index)
            return doc_ids

    def delete_collection(self, collection: str) -> None:
        """Remove every document, chunk and the index of a collection."""
        with self._lock:
            with session_scope() as session:
                session.execute(delete(Chunk).where(Chunk.collection == collection))
                session.execute(delete(Document).where(Document.collection == collection))
            self._indexes.pop(collection, None)
            for path in self._paths(collection):
                if path.exists():
                    path.unlink()

    def search(self, collection: str, query: str, k: int) -> list[Passage]:
        """Top-k passages for a query. Raises if the collection is empty."""
        index = self._load(collection)
        if index is None or index.ntotal == 0:
            raise LookupError(
                f"Knowledge-base collection '{collection}' is empty. Upload documents with POST /documents "
                "or build a dataset knowledge base (see README)."
            )
        vector = self.embed([query])
        scores, ids = index.search(vector, min(k, index.ntotal))
        id_list = [int(i) for i in ids[0] if i != -1]
        with session_scope() as session:
            rows = session.execute(
                select(Chunk.id, Chunk.text, Chunk.chunk_index, Document.id, Document.title, Document.source,
                       Document.created_at, Document.content_date)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.id.in_(id_list))
            ).all()
        by_id = {r[0]: r for r in rows}
        passages: list[Passage] = []
        for score, cid in zip(scores[0], ids[0]):
            row = by_id.get(int(cid))
            if row is None:
                continue
            passages.append(
                Passage(
                    chunk_id=row[0],
                    source_id=f"doc{row[3]}#c{row[2]}",
                    title=row[4],
                    text=row[1],
                    score=float(score),
                    source=row[5],
                    added_at=row[6].isoformat() if row[6] else None,
                    last_modified=row[7],
                )
            )
        return passages


_KB: KnowledgeBase | None = None
_KB_LOCK = threading.Lock()


def get_knowledge_base() -> KnowledgeBase:
    """Process-wide knowledge base singleton."""
    global _KB
    with _KB_LOCK:
        if _KB is None:
            _KB = KnowledgeBase()
        return _KB
