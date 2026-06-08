# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Embedding Engine — Sentence-transformer based vector generation
# =============================================================================

import numpy as np
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_SEQ_LENGTH

console = Console()


class EmbeddingEngine:
    """
    Generates dense vector embeddings using sentence-transformers.
    Supports batch processing with progress tracking.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        """Load the sentence-transformer model."""
        console.print(f"[cyan]📦 Loading embedding model: {self.model_name}...[/cyan]")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

            # Get embedding dimension
            test_emb = self.model.encode(["test"], show_progress_bar=False)
            self.embedding_dim = test_emb.shape[1]

            console.print(f"[green]✓ Model loaded: {self.model_name} "
                          f"(dim={self.embedding_dim})[/green]")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    def encode(self, texts: list[str], batch_size: Optional[int] = None,
               show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (default from config)
            show_progress: Whether to show a progress bar

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or EMBEDDING_BATCH_SIZE

        # Clean texts — handle None/empty
        cleaned = []
        for t in texts:
            if t is None or not str(t).strip():
                cleaned.append("unknown")
            else:
                # Truncate very long texts
                text = str(t).strip()
                if len(text) > 5000:
                    text = text[:5000]
                cleaned.append(text)

        console.print(f"[cyan]🔢 Encoding {len(cleaned)} texts...[/cyan]")

        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        console.print(f"[green]✓ Generated {embeddings.shape[0]} embeddings "
                       f"(shape: {embeddings.shape})[/green]")

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], show_progress=False)[0]

    def cosine_similarity(self, query_embedding: np.ndarray,
                          corpus_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and corpus of embeddings.
        Embeddings are assumed to be L2-normalized.

        Args:
            query_embedding: Single embedding vector (1D)
            corpus_embeddings: Matrix of corpus embeddings (2D)

        Returns:
            Array of similarity scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Since embeddings are normalized, dot product = cosine similarity
        similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
        return similarities

    def find_top_k(self, query_embedding: np.ndarray,
                   corpus_embeddings: np.ndarray,
                   k: int = 10) -> list[tuple[int, float]]:
        """
        Find top-k most similar embeddings to the query.

        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        similarities = self.cosine_similarity(query_embedding, corpus_embeddings)
        top_indices = np.argsort(similarities)[::-1][:k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
