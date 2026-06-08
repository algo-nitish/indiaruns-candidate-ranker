# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Skills Scorer — Weighted skill matching with fuzzy matching
# =============================================================================

import numpy as np
from rapidfuzz import fuzz
from rich.console import Console
from src.parsers.jd_parser import ParsedJobDescription
from src.parsers.candidate_parser import CandidateProfile
from config.settings import (
    SKILL_SIMILARITY_THRESHOLD,
    MUST_HAVE_WEIGHT,
    NICE_TO_HAVE_WEIGHT,
)

console = Console()

# Common skill synonyms for better matching
SKILL_SYNONYMS = {
    "javascript": ["js", "es6", "es2015", "ecmascript"],
    "typescript": ["ts"],
    "python": ["py", "python3", "python2"],
    "react": ["reactjs", "react.js", "react js"],
    "angular": ["angularjs", "angular.js", "angular 2+"],
    "vue": ["vuejs", "vue.js", "vue 3"],
    "node": ["nodejs", "node.js", "node js"],
    "express": ["expressjs", "express.js"],
    "mongodb": ["mongo", "mongo db"],
    "postgresql": ["postgres", "pg", "psql"],
    "mysql": ["my sql"],
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["dl", "deep-learning"],
    "natural language processing": ["nlp"],
    "computer vision": ["cv"],
    "artificial intelligence": ["ai"],
    "data science": ["ds"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp", "google cloud"],
    "microsoft azure": ["azure"],
    "docker": ["containerization", "containers"],
    "kubernetes": ["k8s"],
    "continuous integration": ["ci", "ci/cd", "cicd"],
    "continuous deployment": ["cd"],
    "rest api": ["restful", "rest", "restful api"],
    "graphql": ["graph ql"],
    "sql": ["structured query language"],
    "nosql": ["no sql", "non-relational"],
    "html": ["html5"],
    "css": ["css3", "cascading style sheets"],
    "sass": ["scss"],
    "git": ["github", "gitlab", "version control"],
    "agile": ["scrum", "kanban", "sprint"],
    "tensorflow": ["tf"],
    "pytorch": ["torch"],
    "pandas": ["pd"],
    "numpy": ["np"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "power bi": ["powerbi"],
    "tableau": ["data visualization"],
    "java": ["j2ee", "jdk"],
    "c#": ["csharp", "c sharp", ".net"],
    "c++": ["cpp", "c plus plus"],
    "go": ["golang"],
    "rust": ["rustlang"],
    "ruby": ["ruby on rails", "ror"],
    "php": ["laravel", "symfony"],
    "swift": ["ios development"],
    "kotlin": ["android development"],
}

# Build reverse mapping for quick lookup
_SYNONYM_REVERSE = {}
for canonical, synonyms in SKILL_SYNONYMS.items():
    for syn in synonyms:
        _SYNONYM_REVERSE[syn] = canonical
    _SYNONYM_REVERSE[canonical] = canonical


class SkillsScorer:
    """
    Scores candidates based on how well their skills match JD requirements.
    Uses fuzzy matching and synonym expansion for robust matching.
    Must-have skills carry higher weight than nice-to-have skills.
    """

    def __init__(self, similarity_threshold: float = SKILL_SIMILARITY_THRESHOLD):
        self.threshold = similarity_threshold

    def score(
        self,
        jd: ParsedJobDescription,
        candidates: list[CandidateProfile],
    ) -> dict[str, float]:
        """
        Compute skills match scores for all candidates.

        Args:
            jd: Parsed job description with must-have and nice-to-have skills
            candidates: List of candidate profiles

        Returns:
            Dict mapping candidate_id → skills_score (0.0 to 1.0)
        """
        console.print("[cyan]🎯 Computing skills match scores...[/cyan]")

        scores = {}
        for candidate in candidates:
            score = self._score_candidate(jd, candidate)
            scores[candidate.candidate_id] = score

        scores_arr = np.array(list(scores.values()))
        console.print(f"[green]✓ Skills scores: "
                       f"min={scores_arr.min():.4f}, "
                       f"max={scores_arr.max():.4f}, "
                       f"mean={scores_arr.mean():.4f}[/green]")

        return scores

    def _score_candidate(self, jd: ParsedJobDescription, candidate: CandidateProfile) -> float:
        """Score a single candidate's skills against JD requirements."""
        candidate_skills = set(self._normalize_skill(s) for s in candidate.skills)

        # Score must-have skills
        must_have_matched = 0
        must_have_total = len(jd.must_have_skills)
        for skill in jd.must_have_skills:
            normalized = self._normalize_skill(skill)
            if self._skill_matches(normalized, candidate_skills):
                must_have_matched += 1

        # Score nice-to-have skills
        nice_matched = 0
        nice_total = len(jd.nice_to_have_skills)
        for skill in jd.nice_to_have_skills:
            normalized = self._normalize_skill(skill)
            if self._skill_matches(normalized, candidate_skills):
                nice_matched += 1

        # Weighted combination
        must_have_score = must_have_matched / max(must_have_total, 1)
        nice_to_have_score = nice_matched / max(nice_total, 1)

        total_weight = MUST_HAVE_WEIGHT + NICE_TO_HAVE_WEIGHT
        final_score = (
            (must_have_score * MUST_HAVE_WEIGHT + nice_to_have_score * NICE_TO_HAVE_WEIGHT)
            / total_weight
        )

        return min(final_score, 1.0)

    def _normalize_skill(self, skill: str) -> str:
        """Normalize a skill name using synonym mapping."""
        skill = skill.lower().strip()
        return _SYNONYM_REVERSE.get(skill, skill)

    def _skill_matches(self, target_skill: str, candidate_skills: set[str]) -> bool:
        """Check if a target skill matches any of the candidate's skills."""
        # Exact match
        if target_skill in candidate_skills:
            return True

        # Fuzzy match against each candidate skill
        for cs in candidate_skills:
            ratio = fuzz.ratio(target_skill, cs) / 100.0
            if ratio >= self.threshold:
                return True

            # Partial match (e.g., "python" matches "python programming")
            partial = fuzz.partial_ratio(target_skill, cs) / 100.0
            if partial >= 0.9:  # Higher threshold for partial matches
                return True

        return False

    def get_skill_details(
        self, jd: ParsedJobDescription, candidate: CandidateProfile
    ) -> dict:
        """Get detailed skill matching breakdown for a candidate."""
        candidate_skills = set(self._normalize_skill(s) for s in candidate.skills)

        must_have_matches = []
        must_have_misses = []
        for skill in jd.must_have_skills:
            normalized = self._normalize_skill(skill)
            if self._skill_matches(normalized, candidate_skills):
                must_have_matches.append(skill)
            else:
                must_have_misses.append(skill)

        nice_matches = []
        nice_misses = []
        for skill in jd.nice_to_have_skills:
            normalized = self._normalize_skill(skill)
            if self._skill_matches(normalized, candidate_skills):
                nice_matches.append(skill)
            else:
                nice_misses.append(skill)

        return {
            "must_have_matched": must_have_matches,
            "must_have_missing": must_have_misses,
            "nice_to_have_matched": nice_matches,
            "nice_to_have_missing": nice_misses,
            "extra_skills": [s for s in candidate.skills
                             if self._normalize_skill(s) not in
                             {self._normalize_skill(s) for s in jd.must_have_skills + jd.nice_to_have_skills}],
        }
