# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Prompt Templates for LLM Operations
# =============================================================================

# =============================================================================
# Job Description Parsing
# =============================================================================
JD_PARSE_PROMPT = """You are an expert recruiter analyzing a job description. Extract structured requirements from the following JD.

---
JOB DESCRIPTION:
{jd_text}
---

Return a JSON object with exactly these fields:
{{
    "role_title": "the job title",
    "role_level": "junior | mid | senior | lead | executive",
    "department": "the department or function",
    "industry": "the industry or domain",
    "must_have_skills": ["list of absolutely required technical skills"],
    "nice_to_have_skills": ["list of preferred but not required skills"],
    "soft_skills": ["list of soft skills and personality traits"],
    "min_experience_years": number or null,
    "max_experience_years": number or null,
    "education_requirements": ["list of education requirements"],
    "key_responsibilities": ["top 5 responsibilities"],
    "role_summary": "a 2-3 sentence summary of what this role truly needs"
}}

Be precise. Distinguish between must-have and nice-to-have skills carefully.
Only return valid JSON, nothing else."""


# =============================================================================
# Candidate Profile Summarization
# =============================================================================
CANDIDATE_SUMMARIZE_PROMPT = """Summarize this candidate's profile into a structured assessment for recruiter review.

---
CANDIDATE PROFILE:
{candidate_text}
---

Return a JSON object:
{{
    "summary": "2-3 sentence professional summary",
    "primary_skills": ["top technical skills"],
    "experience_years": number,
    "experience_domains": ["industries/domains worked in"],
    "career_trajectory": "ascending | lateral | descending | early-career",
    "strengths": ["top 3 strengths relevant to hiring"],
    "potential_gaps": ["any notable gaps or concerns"]
}}

Only return valid JSON, nothing else."""


# =============================================================================
# LLM Re-Ranking (evaluate a candidate against a JD)
# =============================================================================
RERANK_PROMPT = """You are a world-class technical recruiter. Evaluate how well this candidate fits the given role.

---
JOB REQUIREMENTS:
{jd_summary}

CANDIDATE PROFILE:
{candidate_summary}

PRE-COMPUTED SCORES:
- Semantic Similarity: {semantic_score:.2f}/1.0
- Skills Match: {skills_score:.2f}/1.0
- Experience Match: {experience_score:.2f}/1.0
---

Consider the FULL picture — not just keyword overlap. Think about:
1. Does their career trajectory suggest they'd succeed in this role?
2. Do their skills truly match what the role needs, or is it surface-level?
3. Are there transferable skills or experiences that make them stronger than scores suggest?
4. Any red flags (job hopping, skill gaps, overqualification)?

Return a JSON object:
{{
    "fit_score": <float between 0.0 and 1.0>,
    "confidence": <float between 0.0 and 1.0>,
    "rationale": "<2-3 sentences explaining why this candidate is/isn't a good fit>",
    "strengths": ["top strengths for this specific role"],
    "concerns": ["any concerns for this specific role"],
    "recommendation": "strong_yes | yes | maybe | no | strong_no"
}}

Only return valid JSON, nothing else."""


# =============================================================================
# Batch Re-Ranking (multiple candidates at once — more efficient)
# =============================================================================
BATCH_RERANK_PROMPT = """You are a world-class technical recruiter. Rank these candidates for the given role.

---
JOB REQUIREMENTS:
{jd_summary}

CANDIDATES (with pre-computed scores):
{candidates_block}
---

For each candidate, evaluate holistic fit considering career trajectory, genuine skill alignment, transferable experience, and potential red flags.

Return a JSON array, one entry per candidate, sorted by fit (best first):
[
    {{
        "candidate_id": "...",
        "fit_score": <0.0-1.0>,
        "rationale": "<1-2 sentences>",
        "recommendation": "strong_yes | yes | maybe | no | strong_no"
    }},
    ...
]

Only return valid JSON, nothing else."""
