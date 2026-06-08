# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Experience Scorer — Career history and experience relevance scoring
# =============================================================================

import numpy as np
from rich.console import Console
from src.parsers.jd_parser import ParsedJobDescription
from src.parsers.candidate_parser import CandidateProfile
from config.settings import EXPERIENCE_YEAR_TOLERANCE, MAX_EXPERIENCE_YEARS

console = Console()

# Role level to expected experience mapping
LEVEL_EXPERIENCE_MAP = {
    "junior": (0, 3),
    "mid": (2, 6),
    "senior": (5, 12),
    "lead": (8, 18),
    "executive": (12, 30),
}


class ExperienceScorer:
    """
    Scores candidates based on experience relevance:
    - Years of experience vs. JD requirements
    - Role level alignment
    - Industry/domain overlap
    """

    def score(
        self,
        jd: ParsedJobDescription,
        candidates: list[CandidateProfile],
    ) -> dict[str, float]:
        """
        Compute experience relevance scores for all candidates.

        Args:
            jd: Parsed job description
            candidates: List of candidate profiles

        Returns:
            Dict mapping candidate_id → experience_score (0.0 to 1.0)
        """
        console.print("[cyan]📊 Computing experience relevance scores...[/cyan]")

        scores = {}
        for candidate in candidates:
            score = self._score_candidate(jd, candidate)
            scores[candidate.candidate_id] = score

        scores_arr = np.array(list(scores.values()))
        console.print(f"[green]✓ Experience scores: "
                       f"min={scores_arr.min():.4f}, "
                       f"max={scores_arr.max():.4f}, "
                       f"mean={scores_arr.mean():.4f}[/green]")

        return scores

    def _score_candidate(self, jd: ParsedJobDescription, candidate: CandidateProfile) -> float:
        """Score a single candidate's experience against JD requirements."""
        sub_scores = []
        weights = []

        # 1. Years of experience match (weight: 40%)
        years_score = self._score_years(jd, candidate)
        sub_scores.append(years_score)
        weights.append(0.40)

        # 2. Role level alignment (weight: 30%)
        level_score = self._score_level(jd, candidate)
        sub_scores.append(level_score)
        weights.append(0.30)

        # 3. Industry/domain overlap (weight: 30%)
        domain_score = self._score_domain(jd, candidate)
        sub_scores.append(domain_score)
        weights.append(0.30)

        # Weighted combination
        final = sum(s * w for s, w in zip(sub_scores, weights))
        return min(max(final, 0.0), 1.0)

    def _score_years(self, jd: ParsedJobDescription, candidate: CandidateProfile) -> float:
        """
        Score based on years of experience.
        Penalizes both under-experience and significant over-experience.
        """
        candidate_years = candidate.experience_years

        # If JD specifies experience range
        min_years = jd.min_experience_years or 0
        max_years = jd.max_experience_years

        if max_years is None:
            # Infer from role level
            level_range = LEVEL_EXPERIENCE_MAP.get(jd.role_level, (0, 30))
            max_years = level_range[1]

        # Perfect range: within specified bounds
        if min_years <= candidate_years <= max_years:
            return 1.0

        # Under-experienced
        if candidate_years < min_years:
            deficit = min_years - candidate_years
            return max(0.0, 1.0 - (deficit / max(EXPERIENCE_YEAR_TOLERANCE * 2, 1)))

        # Over-experienced (mild penalty — experience is generally good)
        if candidate_years > max_years:
            surplus = candidate_years - max_years
            # Gentle penalty — being over-experienced is better than under
            return max(0.3, 1.0 - (surplus / (MAX_EXPERIENCE_YEARS * 0.5)))

        return 0.5

    def _score_level(self, jd: ParsedJobDescription, candidate: CandidateProfile) -> float:
        """Score based on role level alignment."""
        # Infer candidate level from experience
        candidate_level = self._infer_level(candidate.experience_years)
        jd_level = jd.role_level.lower()

        levels = ["junior", "mid", "senior", "lead", "executive"]
        try:
            jd_idx = levels.index(jd_level)
        except ValueError:
            return 0.5  # Unknown level, neutral score

        try:
            cand_idx = levels.index(candidate_level)
        except ValueError:
            return 0.5

        # Perfect match
        if jd_idx == cand_idx:
            return 1.0

        # One level off
        diff = abs(jd_idx - cand_idx)
        if diff == 1:
            return 0.7

        # Two+ levels off
        return max(0.1, 1.0 - (diff * 0.3))

    def _score_domain(self, jd: ParsedJobDescription, candidate: CandidateProfile) -> float:
        """Score based on industry/domain overlap."""
        if not jd.industry:
            return 0.5  # No domain specified, neutral

        jd_industry = jd.industry.lower()
        candidate_industries = [ind.lower() for ind in candidate.industries]

        # Direct match
        for ind in candidate_industries:
            if jd_industry in ind or ind in jd_industry:
                return 1.0

        # Check title for domain hints
        title = candidate.current_title.lower()
        if jd_industry in title:
            return 0.8

        # No domain match — not necessarily bad (transferable skills)
        if candidate_industries:
            return 0.3  # Has experience but different domain
        return 0.4  # No domain info available

    def _infer_level(self, years: float) -> str:
        """Infer career level from years of experience."""
        if years < 2:
            return "junior"
        elif years < 5:
            return "mid"
        elif years < 10:
            return "senior"
        elif years < 15:
            return "lead"
        else:
            return "executive"
