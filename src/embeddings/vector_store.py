# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Vector Store — FAISS-based index for fast similarity search
# =============================================================================

import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()


class VectorStore:
    """
    FAISS-based vector store for efficient nearest neighbor search.
    Supports save/load for persistence.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.metadata = []  # Store associated metadata (candidate IDs, etc.)
        self._init_index()

    def _init_index(self):
        """Initialize the FAISS index."""
        try:
            import faiss
            # Use L2 (Inner Product with normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
            console.print(f"[green]✓ FAISS index initialized (dim={self.dimension})[/green]")
        except ImportError:
            console.print("[yellow]⚠ FAISS not available, falling back to numpy search[/yellow]")
            self.index = None
            self._embeddings = None

    def add(self, embeddings: np.ndarray, metadata: Optional[list] = None):
        """
        Add embeddings to the index.

        Args:
            embeddings: numpy array of shape (n, dimension), must be float32
            metadata: Optional list of associated metadata (same length as embeddings)
        """
        embeddings = embeddings.astype(np.float32)

        if self.index is not None:
            import faiss
            # Normalize for inner product = cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            # Numpy fallback
            if self._embeddings is None:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])

        if metadata:
            self.metadata.extend(metadata)

        console.print(f"[green]✓ Added {len(embeddings)} vectors to store "
                       f"(total: {self.size})[/green]")

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float, dict]]:
        """
        Search for the k nearest neighbors.

        Args:
            query: Query embedding vector (1D or 2D)
            k: Number of results to return

        Returns:
            List of (index, score, metadata) tuples
        """
        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.index is not None:
            import faiss
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, min(k, self.size))
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            normalized = self._embeddings / (norms + 1e-10)
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            scores = np.dot(normalized, query_norm.T).flatten()
            indices = np.argsort(scores)[::-1][:k]
            scores = scores[indices]

        results = []
        for idx, score in zip(indices, scores):
            idx = int(idx)
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append((idx, float(score), meta))

        return results

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        if self.index is not None:
            return self.index.ntotal
        elif self._embeddings is not None:
            return len(self._embeddings)
        return 0

    def save(self, filepath: str | Path):
        """Save the vector store to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dimension": self.dimension,
            "metadata": self.metadata,
        }

        if self.index is not None:
            import faiss
            index_path = str(filepath) + ".faiss"
            faiss.write_index(self.index, index_path)
            data["index_path"] = index_path
        else:
            data["embeddings"] = self._embeddings

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        console.print(f"[green]✓ Vector store saved to {filepath}[/green]")

    @classmethod
    def load(cls, filepath: str | Path) -> "VectorStore":
        """Load a vector store from disk."""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        store = cls(data["dimension"])
        store.metadata = data["metadata"]

        if "index_path" in data:
            import faiss
            store.index = faiss.read_index(data["index_path"])
        elif "embeddings" in data:
            store._embeddings = data["embeddings"]

        console.print(f"[green]✓ Vector store loaded from {filepath} "
                       f"({store.size} vectors)[/green]")
        return store
