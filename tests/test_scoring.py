# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Tests — Scoring and Honeypot Detection
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.scoring.candidate_scorer import CandidateScorer
from src.scoring.honeypot_detector import HoneypotDetector


class TestCandidateScorer:
    """Test the candidate scoring logic."""

    def setup_method(self):
        self.scorer = CandidateScorer()
        self.detector = HoneypotDetector()

    def test_skills_scoring(self):
        """Test scoring of skills match."""
        # Candidate with perfect matching skills
        c_good = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "current_title": "Senior AI Engineer",
                "years_of_experience": 6.0,
                "current_company": "Redrob",
                "current_industry": "AI",
            },
            "skills": [
                {"name": "sentence-transformers", "proficiency": "expert", "duration_months": 24, "endorsements": 15},
                {"name": "faiss", "proficiency": "expert", "duration_months": 24, "endorsements": 15},
                {"name": "python", "proficiency": "expert", "duration_months": 48, "endorsements": 20},
            ],
            "redrob_signals": {
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 10.0,
            }
        }
        scores = self.scorer.score(c_good)
        assert scores["skills_match"] > 0.1

    def test_honeypot_detection(self):
        """Test that honeypot detection flags impossible profiles."""
        # Candidate claiming to be expert in many skills with 0 duration
        c_honeypot = {
            "candidate_id": "CAND_9999999",
            "profile": {
                "current_title": "Marketing Manager",
                "years_of_experience": 2.0,
            },
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 50},
                {"name": "TensorFlow", "proficiency": "expert", "duration_months": 0, "endorsements": 50},
                {"name": "PyTorch", "proficiency": "expert", "duration_months": 0, "endorsements": 50},
                {"name": "Docker", "proficiency": "expert", "duration_months": 0, "endorsements": 50},
            ],
            "redrob_signals": {
                "github_activity_score": -1,
            }
        }
        assert self.detector.is_honeypot(c_honeypot) is True
