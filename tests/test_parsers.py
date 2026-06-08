# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Tests — Parsers
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from src.parsers.candidate_parser import CandidateParser, CandidateProfile


class TestCandidateParser:
    """Test the candidate profile parser."""

    def setup_method(self):
        self.parser = CandidateParser()

    def test_parse_basic_dataframe(self):
        """Test parsing a simple DataFrame with standard columns."""
        df = pd.DataFrame({
            "candidate_id": ["C001", "C002", "C003"],
            "name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
            "title": ["Senior Python Dev", "ML Engineer", "Data Scientist"],
            "company": ["Google", "Meta", "Amazon"],
            "experience": [8, 5, 3],
            "skills": [
                "Python, Django, AWS, PostgreSQL",
                "Python, PyTorch, TensorFlow, MLOps",
                "Python, Pandas, SQL, Tableau",
            ],
            "education": [
                "M.S. Computer Science",
                "Ph.D. Machine Learning",
                "B.Tech Information Technology",
            ],
        })

        profiles = self.parser.parse_dataframe(df)

        assert len(profiles) == 3
        assert profiles[0].candidate_id == "C001"
        assert profiles[0].name == "Alice Smith"
        assert profiles[0].experience_years == 8.0
        assert "python" in profiles[0].skills
        assert "django" in profiles[0].skills

    def test_parse_messy_column_names(self):
        """Test that the parser handles various column naming conventions."""
        df = pd.DataFrame({
            "Full_Name": ["Test User"],
            "Years_of_Experience": ["5"],
            "Key_Skills": ["Java, Spring Boot, Microservices"],
            "Current_Title": ["Backend Developer"],
        })

        profiles = self.parser.parse_dataframe(df)
        assert len(profiles) == 1
        # Should still extract skills even with different column name
        assert profiles[0].current_title == "Backend Developer"

    def test_parse_experience_formats(self):
        """Test parsing various experience formats."""
        parser = CandidateParser()

        assert parser._parse_experience("5") == 5.0
        assert parser._parse_experience("5.5") == 5.5
        assert parser._parse_experience("5 years") == 5.0
        assert parser._parse_experience("3-5 years") == 4.0
        assert parser._parse_experience("") == 0.0

    def test_parse_list_field(self):
        """Test parsing comma/pipe/semicolon separated lists."""
        parser = CandidateParser()

        assert parser._parse_list_field("Python, Java, Go") == ["Python", "Java", "Go"]
        assert parser._parse_list_field("Python|Java|Go") == ["Python", "Java", "Go"]
        assert parser._parse_list_field("Python;Java;Go") == ["Python", "Java", "Go"]
        assert parser._parse_list_field("") == []

    def test_summary_text_generation(self):
        """Test that summary text is generated for embedding."""
        df = pd.DataFrame({
            "name": ["Test User"],
            "title": ["Python Developer"],
            "skills": ["Python, Flask"],
            "experience": [3],
        })

        profiles = self.parser.parse_dataframe(df)
        assert len(profiles[0].summary_text) > 0
        assert "Python" in profiles[0].summary_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
