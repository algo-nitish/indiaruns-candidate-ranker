# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Hybrid Ranker — Multi-signal fusion with LLM re-ranking
# =============================================================================

import numpy as np
from typing import Optional
from rich.console import Console
from rich.table import Table

from src.parsers.jd_parser import ParsedJobDescription, JDParser
from src.parsers.candidate_parser import CandidateProfile
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.semantic_scorer import SemanticScorer
from src.scoring.skills_scorer import SkillsScorer
from src.scoring.experience_scorer import ExperienceScorer
from src.scoring.behavioral_scorer import BehavioralScorer
from src.llm.llm_client import LLMClient
from src.llm.prompts import RERANK_PROMPT, BATCH_RERANK_PROMPT
from config.settings import SCORING_WEIGHTS, LLM_RERANK_TOP_N, OUTPUT_TOP_N

console = Console()


class HybridRanker:
    """
    The core ranking engine that combines multiple scoring signals:

    1. Semantic Similarity (30%) — Embedding-based cosine similarity
    2. Skills Match (25%) — Weighted skill overlap with fuzzy matching
    3. Experience Relevance (20%) — Years + domain + level alignment
    4. Behavioral Signals (10%) — Platform activity and engagement
    5. LLM Re-Ranking (15%) — GPT/Gemini evaluates top candidates

    Produces a final ranked list with composite scores and rationale.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        llm_client: Optional[LLMClient] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.embedding_engine = embedding_engine
        self.llm_client = llm_client
        self.weights = weights or SCORING_WEIGHTS

        # Initialize sub-scorers
        self.semantic_scorer = SemanticScorer(embedding_engine)
        self.skills_scorer = SkillsScorer()
        self.experience_scorer = ExperienceScorer()
        self.behavioral_scorer = BehavioralScorer()

    def rank(
        self,
        jd: ParsedJobDescription,
        candidates: list[CandidateProfile],
        top_n: int = OUTPUT_TOP_N,
        use_llm_rerank: bool = True,
    ) -> list[dict]:
        """
        Rank candidates against a job description using hybrid scoring.

        Args:
            jd: Parsed job description
            candidates: List of candidate profiles
            top_n: Number of top candidates to return
            use_llm_rerank: Whether to use LLM for final re-ranking

        Returns:
            List of ranked candidate dicts with scores and rationale
        """
        console.print("\n[bold magenta]"
                       "═══════════════════════════════════════════\n"
                       "  🚀 HYBRID RANKING ENGINE\n"
                       "═══════════════════════════════════════════"
                       "[/bold magenta]\n")

        console.print(f"[bold]Ranking {len(candidates)} candidates for: "
                       f"{jd.role_title}[/bold]\n")

        # -----------------------------------------------------------------
        # Step 1: Compute individual scores
        # -----------------------------------------------------------------
        jd_parser = JDParser.__new__(JDParser)  # Just for utility methods
        jd_text = jd_parser.get_embedding_text(jd)
        candidate_texts = [c.summary_text for c in candidates]
        candidate_ids = [c.candidate_id for c in candidates]

        # Semantic scores
        semantic_scores = self.semantic_scorer.score(jd_text, candidate_texts, candidate_ids)

        # Skills scores
        skills_scores = self.skills_scorer.score(jd, candidates)

        # Experience scores
        experience_scores = self.experience_scorer.score(jd, candidates)

        # Behavioral scores
        behavioral_scores = self.behavioral_scorer.score(candidates)

        # -----------------------------------------------------------------
        # Step 2: Compute pre-LLM composite scores
        # -----------------------------------------------------------------
        pre_llm_weight = (
            self.weights["semantic"]
            + self.weights["skills"]
            + self.weights["experience"]
            + self.weights["behavioral"]
        )

        composite_scores = {}
        for cid in candidate_ids:
            weighted_sum = (
                semantic_scores.get(cid, 0) * self.weights["semantic"]
                + skills_scores.get(cid, 0) * self.weights["skills"]
                + experience_scores.get(cid, 0) * self.weights["experience"]
                + behavioral_scores.get(cid, 0) * self.weights["behavioral"]
            )
            composite_scores[cid] = weighted_sum / pre_llm_weight

        # Sort by composite score
        sorted_ids = sorted(composite_scores, key=composite_scores.get, reverse=True)

        # -----------------------------------------------------------------
        # Step 3: LLM Re-Ranking (top-N candidates only)
        # -----------------------------------------------------------------
        llm_scores = {}
        llm_rationales = {}
        llm_recommendations = {}

        if use_llm_rerank and self.llm_client:
            top_for_rerank = sorted_ids[:LLM_RERANK_TOP_N]
            candidates_map = {c.candidate_id: c for c in candidates}

            console.print(f"\n[cyan]🧠 LLM re-ranking top {len(top_for_rerank)} candidates...[/cyan]")

            for cid in top_for_rerank:
                candidate = candidates_map[cid]
                try:
                    result = self._llm_rerank_single(
                        jd, candidate,
                        semantic_scores[cid],
                        skills_scores[cid],
                        experience_scores[cid],
                    )
                    llm_scores[cid] = result.get("fit_score", 0.5)
                    llm_rationales[cid] = result.get("rationale", "")
                    llm_recommendations[cid] = result.get("recommendation", "maybe")
                except Exception as e:
                    console.print(f"[yellow]⚠ LLM re-rank failed for {cid}: {e}[/yellow]")
                    llm_scores[cid] = composite_scores[cid]
                    llm_rationales[cid] = "LLM evaluation unavailable"
                    llm_recommendations[cid] = "maybe"

            # Fill remaining with neutral LLM scores
            for cid in sorted_ids:
                if cid not in llm_scores:
                    llm_scores[cid] = 0.0
                    llm_rationales[cid] = "Not evaluated by LLM (outside top-N)"
                    llm_recommendations[cid] = "not_evaluated"
        else:
            # No LLM re-ranking
            for cid in sorted_ids:
                llm_scores[cid] = 0.0
                llm_rationales[cid] = "LLM re-ranking disabled"
                llm_recommendations[cid] = "not_evaluated"

        # -----------------------------------------------------------------
        # Step 4: Final composite with LLM scores
        # -----------------------------------------------------------------
        final_scores = {}
        for cid in candidate_ids:
            if use_llm_rerank and self.llm_client and cid in llm_scores and llm_scores[cid] > 0:
                final = (
                    semantic_scores.get(cid, 0) * self.weights["semantic"]
                    + skills_scores.get(cid, 0) * self.weights["skills"]
                    + experience_scores.get(cid, 0) * self.weights["experience"]
                    + behavioral_scores.get(cid, 0) * self.weights["behavioral"]
                    + llm_scores.get(cid, 0) * self.weights["llm_rerank"]
                )
            else:
                # Without LLM, redistribute weight
                final = composite_scores[cid]
            final_scores[cid] = final

        # Final sort
        final_sorted = sorted(final_scores, key=final_scores.get, reverse=True)

        # -----------------------------------------------------------------
        # Step 5: Build output
        # -----------------------------------------------------------------
        candidates_map = {c.candidate_id: c for c in candidates}
        results = []

        for rank, cid in enumerate(final_sorted[:top_n], start=1):
            candidate = candidates_map[cid]
            results.append({
                "rank": rank,
                "candidate_id": cid,
                "name": candidate.name,
                "current_title": candidate.current_title,
                "composite_score": round(final_scores[cid], 4),
                "semantic_score": round(semantic_scores.get(cid, 0), 4),
                "skills_score": round(skills_scores.get(cid, 0), 4),
                "experience_score": round(experience_scores.get(cid, 0), 4),
                "behavioral_score": round(behavioral_scores.get(cid, 0), 4),
                "llm_score": round(llm_scores.get(cid, 0), 4),
                "recommendation": llm_recommendations.get(cid, "not_evaluated"),
                "rationale": llm_rationales.get(cid, ""),
                "top_skills": candidate.skills[:10],
                "experience_years": candidate.experience_years,
            })

        # Print top results
        self._print_results(results[:15])

        return results

    def _llm_rerank_single(
        self,
        jd: ParsedJobDescription,
        candidate: CandidateProfile,
        semantic_score: float,
        skills_score: float,
        experience_score: float,
    ) -> dict:
        """Re-rank a single candidate using LLM."""
        prompt = RERANK_PROMPT.format(
            jd_summary=f"{jd.role_title} | {jd.role_summary} | "
                        f"Must-have: {', '.join(jd.must_have_skills[:10])} | "
                        f"Nice-to-have: {', '.join(jd.nice_to_have_skills[:5])}",
            candidate_summary=candidate.summary_text[:1500],
            semantic_score=semantic_score,
            skills_score=skills_score,
            experience_score=experience_score,
        )
        return self.llm_client.generate_json(prompt)

    def _print_results(self, results: list[dict]):
        """Print a formatted table of top ranked candidates."""
        console.print("\n[bold green]"
                       "═══════════════════════════════════════════\n"
                       "  🏆 TOP RANKED CANDIDATES\n"
                       "═══════════════════════════════════════════"
                       "[/bold green]\n")

        table = Table(title="Candidate Rankings", show_lines=True)
        table.add_column("#", style="bold", justify="right", width=4)
        table.add_column("Name", style="bold cyan", min_width=15)
        table.add_column("Title", min_width=20)
        table.add_column("Score", justify="center", style="bold")
        table.add_column("Sem", justify="center")
        table.add_column("Skill", justify="center")
        table.add_column("Exp", justify="center")
        table.add_column("Beh", justify="center")
        table.add_column("LLM", justify="center")
        table.add_column("Rec", style="bold")

        for r in results:
            # Color code recommendation
            rec = r.get("recommendation", "?")
            rec_style = {
                "strong_yes": "[bold green]",
                "yes": "[green]",
                "maybe": "[yellow]",
                "no": "[red]",
                "strong_no": "[bold red]",
            }.get(rec, "[dim]")

            table.add_row(
                str(r["rank"]),
                r["name"][:20] if r["name"] else "—",
                r["current_title"][:25] if r["current_title"] else "—",
                f"{r['composite_score']:.3f}",
                f"{r['semantic_score']:.2f}",
                f"{r['skills_score']:.2f}",
                f"{r['experience_score']:.2f}",
                f"{r['behavioral_score']:.2f}",
                f"{r['llm_score']:.2f}",
                f"{rec_style}{rec}[/]",
            )

        console.print(table)
