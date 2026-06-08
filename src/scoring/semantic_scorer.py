# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Semantic Scorer — Embedding-based cosine similarity scoring
# =============================================================================

import numpy as np
from rich.console import Console
from src.embeddings.embedding_engine import EmbeddingEngine

console = Console()


class SemanticScorer:
    """
    Scores candidates based on semantic similarity between their profile
    and the job description using dense vector embeddings.

    This captures meaning beyond keywords — e.g., "React developer" and
    "frontend engineer with React.js" will score high despite different words.
    """

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.engine = embedding_engine

    def score(
        self,
        jd_text: str,
        candidate_texts: list[str],
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """
        Compute semantic similarity scores for candidates against a JD.

        Args:
            jd_text: Rich text representation of the job description
            candidate_texts: List of candidate summary texts
            candidate_ids: List of candidate identifiers

        Returns:
            Dict mapping candidate_id → semantic_score (0.0 to 1.0)
        """
        console.print("[cyan]📐 Computing semantic similarity scores...[/cyan]")

        # Encode JD
        jd_embedding = self.engine.encode_single(jd_text)

        # Encode all candidates
        candidate_embeddings = self.engine.encode(candidate_texts)

        # Compute cosine similarities
        similarities = self.engine.cosine_similarity(jd_embedding, candidate_embeddings)

        # Normalize to 0-1 range (cosine similarity of normalized vectors is already in [-1, 1])
        # Clamp to [0, 1] since negative similarity means anti-correlated (not useful here)
        similarities = np.clip(similarities, 0.0, 1.0)

        scores = {}
        for cid, sim in zip(candidate_ids, similarities):
            scores[cid] = float(sim)

        # Report statistics
        scores_arr = np.array(list(scores.values()))
        console.print(f"[green]✓ Semantic scores: "
                       f"min={scores_arr.min():.4f}, "
                       f"max={scores_arr.max():.4f}, "
                       f"mean={scores_arr.mean():.4f}[/green]")

        return scores
