# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Candidate Parser — Profile structuring and feature extraction
# =============================================================================

import re
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class CandidateProfile(BaseModel):
    """Structured representation of a candidate's profile."""
    candidate_id: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""

    # Professional
    current_title: str = ""
    current_company: str = ""
    experience_years: float = 0.0
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    # Career history
    previous_roles: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)

    # Behavioral signals
    linkedin_activity: Optional[float] = None  # Engagement score
    github_repos: Optional[int] = None
    platform_score: Optional[float] = None

    # Text representations
    summary_text: str = ""       # For embedding
    raw_data: dict = Field(default_factory=dict)  # Original row data


class CandidateParser:
    """
    Parses candidate profiles from dataset into structured CandidateProfile objects.
    Handles varying column names and messy data gracefully.
    """

    # Common column name mappings (lowercase)
    COLUMN_MAPS = {
        "id": ["candidate_id", "id", "applicant_id", "profile_id", "user_id", "sl no", "s.no", "sno"],
        "name": ["name", "full_name", "candidate_name", "fullname", "first_name"],
        "email": ["email", "email_address", "e-mail", "mail"],
        "phone": ["phone", "phone_number", "mobile", "contact", "contact_number"],
        "title": ["title", "current_title", "job_title", "designation", "position", "current_role", "role"],
        "company": ["company", "current_company", "organization", "employer", "current_employer"],
        "experience": ["experience", "experience_years", "years_of_experience", "total_experience",
                        "yoe", "exp", "work_experience", "years_experience"],
        "skills": ["skills", "skill_set", "technical_skills", "key_skills", "competencies",
                    "core_skills", "primary_skills"],
        "education": ["education", "degree", "qualification", "educational_qualification",
                       "highest_education", "academic"],
        "summary": ["summary", "about", "bio", "profile_summary", "professional_summary",
                     "description", "about_me", "overview"],
        "location": ["location", "city", "address", "region", "current_location"],
        "linkedin": ["linkedin", "linkedin_url", "linkedin_profile", "linkedin_activity"],
        "github": ["github", "github_url", "github_profile", "github_repos"],
        "certifications": ["certifications", "certificates", "certification"],
        "industries": ["industry", "industries", "domain", "sector"],
        "previous_roles": ["previous_roles", "work_history", "experience_details",
                            "career_history", "past_roles"],
    }

    def __init__(self):
        self._column_mapping = {}  # Resolved column mapping for this dataset

    def parse_dataframe(self, df: pd.DataFrame) -> list[CandidateProfile]:
        """
        Parse an entire DataFrame of candidates into CandidateProfile objects.

        Args:
            df: DataFrame with candidate data

        Returns:
            List of structured CandidateProfile objects
        """
        console.print(f"[bold cyan]👥 Parsing {len(df)} candidate profiles...[/bold cyan]")

        # Resolve column mapping
        self._resolve_columns(df)

        profiles = []
        for idx, row in df.iterrows():
            try:
                profile = self._parse_row(row, idx)
                profiles.append(profile)
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to parse row {idx}: {e}[/yellow]")

        console.print(f"[green]✓ Successfully parsed {len(profiles)}/{len(df)} profiles[/green]")
        return profiles

    def _resolve_columns(self, df: pd.DataFrame):
        """Map dataset columns to our expected fields."""
        df_cols_lower = {col.lower().strip(): col for col in df.columns}

        self._column_mapping = {}
        for field, candidates in self.COLUMN_MAPS.items():
            for candidate in candidates:
                if candidate in df_cols_lower:
                    self._column_mapping[field] = df_cols_lower[candidate]
                    break

        # Report mapping
        console.print("\n[bold]📋 Column Mapping:[/bold]")
        for field, col in self._column_mapping.items():
            console.print(f"   {field:15s} → {col}")

        unmapped_cols = set(df.columns) - set(self._column_mapping.values())
        if unmapped_cols:
            console.print(f"\n[dim]   Unmapped columns: {', '.join(sorted(unmapped_cols))}[/dim]")

    def _get_field(self, row: pd.Series, field: str, default="") -> str:
        """Safely get a field value from a row using resolved mapping."""
        col = self._column_mapping.get(field)
        if col is None:
            return default
        val = row.get(col, default)
        if pd.isna(val):
            return default
        return str(val).strip()

    def _parse_row(self, row: pd.Series, idx: int) -> CandidateProfile:
        """Parse a single row into a CandidateProfile."""
        # Extract skills (handle comma-separated, pipe-separated, etc.)
        skills_raw = self._get_field(row, "skills")
        skills = self._parse_list_field(skills_raw)

        # Extract education
        education_raw = self._get_field(row, "education")
        education = self._parse_list_field(education_raw) if education_raw else []

        # Extract certifications
        certs_raw = self._get_field(row, "certifications")
        certifications = self._parse_list_field(certs_raw) if certs_raw else []

        # Extract experience years (handle various formats)
        experience = self._parse_experience(self._get_field(row, "experience"))

        # Extract previous roles
        prev_roles_raw = self._get_field(row, "previous_roles")
        previous_roles = self._parse_list_field(prev_roles_raw) if prev_roles_raw else []

        # Extract industries
        industries_raw = self._get_field(row, "industries")
        industries = self._parse_list_field(industries_raw) if industries_raw else []

        # Build candidate ID
        candidate_id = self._get_field(row, "id") or str(idx)

        # Build summary text for embedding
        summary_text = self._build_summary_text(row)

        return CandidateProfile(
            candidate_id=candidate_id,
            name=self._get_field(row, "name"),
            email=self._get_field(row, "email"),
            phone=self._get_field(row, "phone"),
            current_title=self._get_field(row, "title"),
            current_company=self._get_field(row, "company"),
            experience_years=experience,
            skills=[s.lower() for s in skills],
            education=education,
            certifications=certifications,
            previous_roles=previous_roles,
            industries=industries,
            linkedin_activity=self._parse_float(self._get_field(row, "linkedin")),
            github_repos=self._parse_int(self._get_field(row, "github")),
            summary_text=summary_text,
            raw_data=row.to_dict(),
        )

    def _parse_list_field(self, text: str) -> list[str]:
        """Parse a text field that may contain a list (comma, pipe, semicolon separated)."""
        if not text:
            return []

        # Try common separators
        for sep in ["|", ";", ","]:
            if sep in text:
                return [item.strip() for item in text.split(sep) if item.strip()]

        # Single value or newline-separated
        items = [line.strip() for line in text.split("\n") if line.strip()]
        if len(items) > 1:
            return items

        return [text.strip()] if text.strip() else []

    def _parse_experience(self, text: str) -> float:
        """Parse experience years from various formats."""
        if not text:
            return 0.0

        # Try direct float conversion
        try:
            return float(text)
        except ValueError:
            pass

        # Extract numbers from text like "5 years", "3-5 years", "5+ years"
        numbers = re.findall(r"(\d+\.?\d*)", text)
        if numbers:
            # If range (e.g., "3-5"), take average
            nums = [float(n) for n in numbers]
            return sum(nums) / len(nums)

        return 0.0

    def _parse_float(self, text: str) -> Optional[float]:
        """Safely parse a float value."""
        if not text:
            return None
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, text: str) -> Optional[int]:
        """Safely parse an integer value."""
        if not text:
            return None
        try:
            return int(float(text))
        except (ValueError, TypeError):
            return None

    def _build_summary_text(self, row: pd.Series) -> str:
        """
        Build a rich text summary of a candidate for embedding generation.
        Combines all available fields into a semantically meaningful passage.
        """
        parts = []

        name = self._get_field(row, "name")
        title = self._get_field(row, "title")
        company = self._get_field(row, "company")
        if name:
            intro = name
            if title:
                intro += f", {title}"
            if company:
                intro += f" at {company}"
            parts.append(intro)

        exp = self._get_field(row, "experience")
        if exp:
            parts.append(f"Experience: {exp}")

        skills = self._get_field(row, "skills")
        if skills:
            parts.append(f"Skills: {skills}")

        education = self._get_field(row, "education")
        if education:
            parts.append(f"Education: {education}")

        summary = self._get_field(row, "summary")
        if summary:
            parts.append(f"About: {summary}")

        prev_roles = self._get_field(row, "previous_roles")
        if prev_roles:
            parts.append(f"Previous roles: {prev_roles}")

        industries = self._get_field(row, "industries")
        if industries:
            parts.append(f"Industries: {industries}")

        certs = self._get_field(row, "certifications")
        if certs:
            parts.append(f"Certifications: {certs}")

        location = self._get_field(row, "location")
        if location:
            parts.append(f"Location: {location}")

        # Fallback: if very little was extracted, use all non-null columns
        if len(parts) < 2:
            for col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{col}: {val}")

        return " | ".join(parts)
