# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Job Description Parser — LLM-powered requirement extraction
# =============================================================================

from typing import Optional
from pydantic import BaseModel, Field
from rich.console import Console
from src.llm.llm_client import LLMClient
from src.llm.prompts import JD_PARSE_PROMPT

console = Console()


class ParsedJobDescription(BaseModel):
    """Structured representation of a parsed job description."""
    role_title: str = ""
    role_level: str = "mid"  # junior | mid | senior | lead | executive
    department: str = ""
    industry: str = ""
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    min_experience_years: Optional[int] = None
    max_experience_years: Optional[int] = None
    education_requirements: list[str] = Field(default_factory=list)
    key_responsibilities: list[str] = Field(default_factory=list)
    role_summary: str = ""
    raw_text: str = ""  # Original JD text


class JDParser:
    """
    Parses job descriptions using LLM to extract structured requirements.
    Goes beyond keyword extraction to understand what the role truly needs.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse(self, jd_text: str) -> ParsedJobDescription:
        """
        Parse a job description text into structured requirements.

        Args:
            jd_text: Raw job description text

        Returns:
            ParsedJobDescription with structured fields
        """
        console.print("[bold cyan]🔍 Parsing job description...[/bold cyan]")

        # Use LLM to extract structured requirements
        prompt = JD_PARSE_PROMPT.format(jd_text=jd_text)
        parsed = self.llm.generate_json(prompt)

        # Build the structured object
        jd = ParsedJobDescription(
            role_title=parsed.get("role_title", ""),
            role_level=parsed.get("role_level", "mid"),
            department=parsed.get("department", ""),
            industry=parsed.get("industry", ""),
            must_have_skills=[s.lower().strip() for s in parsed.get("must_have_skills", [])],
            nice_to_have_skills=[s.lower().strip() for s in parsed.get("nice_to_have_skills", [])],
            soft_skills=[s.lower().strip() for s in parsed.get("soft_skills", [])],
            min_experience_years=parsed.get("min_experience_years"),
            max_experience_years=parsed.get("max_experience_years"),
            education_requirements=parsed.get("education_requirements", []),
            key_responsibilities=parsed.get("key_responsibilities", []),
            role_summary=parsed.get("role_summary", ""),
            raw_text=jd_text,
        )

        self._print_summary(jd)
        return jd

    def _print_summary(self, jd: ParsedJobDescription):
        """Print a readable summary of parsed JD."""
        console.print(f"\n[bold green]✓ Job Description Parsed:[/bold green]")
        console.print(f"  Role: [bold]{jd.role_title}[/bold] ({jd.role_level})")
        console.print(f"  Industry: {jd.industry}")

        if jd.min_experience_years or jd.max_experience_years:
            exp = f"{jd.min_experience_years or 0}–{jd.max_experience_years or '?'} years"
            console.print(f"  Experience: {exp}")

        console.print(f"  Must-have skills ({len(jd.must_have_skills)}): "
                       f"{', '.join(jd.must_have_skills[:8])}"
                       f"{'...' if len(jd.must_have_skills) > 8 else ''}")
        console.print(f"  Nice-to-have skills ({len(jd.nice_to_have_skills)}): "
                       f"{', '.join(jd.nice_to_have_skills[:5])}"
                       f"{'...' if len(jd.nice_to_have_skills) > 5 else ''}")
        console.print(f"  Summary: {jd.role_summary[:120]}...")

    def get_all_skills(self, jd: ParsedJobDescription) -> list[str]:
        """Get combined list of all skills from a parsed JD."""
        return jd.must_have_skills + jd.nice_to_have_skills + jd.soft_skills

    def get_embedding_text(self, jd: ParsedJobDescription) -> str:
        """
        Generate a rich text representation of the JD for embedding.
        Combines key fields into a semantically meaningful passage.
        """
        parts = [
            f"Role: {jd.role_title} ({jd.role_level})",
            f"Industry: {jd.industry}",
            f"Summary: {jd.role_summary}",
            f"Required skills: {', '.join(jd.must_have_skills)}",
            f"Preferred skills: {', '.join(jd.nice_to_have_skills)}",
            f"Responsibilities: {'; '.join(jd.key_responsibilities[:5])}",
        ]
        return " | ".join(parts)
