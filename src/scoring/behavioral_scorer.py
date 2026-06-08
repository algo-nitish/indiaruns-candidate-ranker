# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Behavioral Scorer — Platform activity and engagement signals
# =============================================================================

import numpy as np
from rich.console import Console
from src.parsers.candidate_parser import CandidateProfile

console = Console()


class BehavioralScorer:
    """
    Scores candidates based on behavioral signals and platform activity:
    - LinkedIn activity/engagement
    - GitHub contributions
    - Profile completeness
    - Any other platform-specific signals in the dataset

    This scorer is designed to be flexible — it will use whatever
    behavioral signals are available in the data.
    """

    def score(self, candidates: list[CandidateProfile]) -> dict[str, float]:
        """
        Compute behavioral/activity scores for all candidates.

        Args:
            candidates: List of candidate profiles

        Returns:
            Dict mapping candidate_id → behavioral_score (0.0 to 1.0)
        """
        console.print("[cyan]📱 Computing behavioral signal scores...[/cyan]")

        raw_scores = {}
        for candidate in candidates:
            raw_scores[candidate.candidate_id] = self._score_candidate(candidate)

        # Normalize scores across all candidates (relative ranking)
        scores = self._normalize_scores(raw_scores)

        scores_arr = np.array(list(scores.values()))
        console.print(f"[green]✓ Behavioral scores: "
                       f"min={scores_arr.min():.4f}, "
                       f"max={scores_arr.max():.4f}, "
                       f"mean={scores_arr.mean():.4f}[/green]")

        return scores

    def _score_candidate(self, candidate: CandidateProfile) -> float:
        """Compute raw behavioral score for a single candidate."""
        signals = []
        weights = []

        # 1. Profile completeness (always available)
        completeness = self._profile_completeness(candidate)
        signals.append(completeness)
        weights.append(0.30)

        # 2. LinkedIn activity (if available)
        if candidate.linkedin_activity is not None:
            signals.append(min(candidate.linkedin_activity, 1.0))
            weights.append(0.30)

        # 3. GitHub repos (if available)
        if candidate.github_repos is not None:
            github_score = min(candidate.github_repos / 20.0, 1.0)  # Cap at 20 repos
            signals.append(github_score)
            weights.append(0.25)

        # 4. Platform score (if available)
        if candidate.platform_score is not None:
            signals.append(min(candidate.platform_score, 1.0))
            weights.append(0.15)

        # 5. Check raw data for additional behavioral signals
        extra_score = self._extract_extra_signals(candidate)
        if extra_score is not None:
            signals.append(extra_score)
            weights.append(0.15)

        if not signals:
            return 0.5  # Neutral if no data

        # Weighted average
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(signals, weights)) / total_weight

    def _profile_completeness(self, candidate: CandidateProfile) -> float:
        """Score how complete a candidate's profile is (shows engagement)."""
        fields = [
            bool(candidate.name),
            bool(candidate.current_title),
            bool(candidate.current_company),
            candidate.experience_years > 0,
            len(candidate.skills) > 0,
            len(candidate.education) > 0,
            bool(candidate.summary_text and len(candidate.summary_text) > 50),
            len(candidate.certifications) > 0,
            len(candidate.industries) > 0,
            bool(candidate.email),
        ]
        return sum(fields) / len(fields)

    def _extract_extra_signals(self, candidate: CandidateProfile) -> float | None:
        """
        Look for additional behavioral signals in raw data.
        Handles common column patterns from recruitment datasets.
        """
        raw = candidate.raw_data
        signal_keys = [
            "activity_score", "engagement_score", "profile_score",
            "social_score", "platform_activity", "online_presence",
            "response_rate", "application_count",
        ]

        for key in signal_keys:
            for raw_key, raw_val in raw.items():
                if key in raw_key.lower():
                    try:
                        val = float(raw_val)
                        # Normalize assuming 0-100 scale if > 1
                        if val > 1:
                            val = val / 100.0
                        return min(max(val, 0.0), 1.0)
                    except (ValueError, TypeError):
                        continue

        return None

    def _normalize_scores(self, raw_scores: dict[str, float]) -> dict[str, float]:
        """Normalize scores to [0, 1] range using min-max scaling."""
        values = list(raw_scores.values())
        if not values:
            return raw_scores

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val < 1e-10:
            # All scores are the same
            return {k: 0.5 for k in raw_scores}

        return {
            k: (v - min_val) / range_val
            for k, v in raw_scores.items()
        }
