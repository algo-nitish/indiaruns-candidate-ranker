# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Central Configuration (Revised for actual challenge constraints)
# =============================================================================

import os
from pathlib import Path

# Redirect HuggingFace caches to D: drive to save C: drive space
os.environ["HF_HOME"] = r"d:\IndiaRuns\.cache\huggingface"

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "required_datasets" / "India_runs_data_and_ai_challenge"
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
JD_FILE = DATA_DIR / "job_description.docx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
PRECOMPUTED_DIR = PROJECT_ROOT / "data" / "precomputed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Embedding Model (for pre-computation)
# =============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = 256

# =============================================================================
# LLM (for pre-computation ONLY — NOT during ranking)
# =============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# =============================================================================
# Scoring Weights
# =============================================================================
# These align with the JD's explicit priorities
SCORING_WEIGHTS = {
    "semantic": 0.25,           # Embedding similarity (JD vs profile)
    "skills_match": 0.20,       # Hard skill overlap (must-have vs nice-to-have)
    "career_fit": 0.25,         # Career history analysis (product co vs consulting)
    "behavioral": 0.15,         # Redrob platform signals
    "anti_patterns": 0.15,      # Negative signals (honeypots, keyword stuffers, etc.)
}

# =============================================================================
# JD Requirements (parsed from job_description.docx)
# These are hardcoded because the JD is static and we can't call LLMs at rank time
# =============================================================================
JD_REQUIREMENTS = {
    "role_title": "Senior AI/ML Engineer",
    "role_level": "senior",
    "industry": "HR-tech / AI / Product",
    "experience_range": (5, 9),  # years

    # ── Skills the JD explicitly requires ──
    "must_have_skills": [
        "embeddings", "sentence-transformers", "openai embeddings", "bge", "e5",
        "vector database", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "faiss",
        "python",
        "ranking evaluation", "ndcg", "mrr", "map", "a/b testing",
        "retrieval", "search", "recommendation",
        "nlp", "information retrieval",
        "production ml", "ml engineering",
    ],

    # ── Skills the JD says are nice-to-have ──
    "nice_to_have_skills": [
        "llm fine-tuning", "lora", "qlora", "peft",
        "learning-to-rank", "xgboost",
        "hr-tech", "recruiting tech", "marketplace",
        "distributed systems", "inference optimization",
        "open-source contributions",
    ],

    # ── Titles that indicate good fit ──
    "positive_titles": [
        "ml engineer", "machine learning engineer", "ai engineer",
        "data scientist", "senior ml engineer", "staff ml engineer",
        "applied scientist", "research engineer", "nlp engineer",
        "search engineer", "ranking engineer", "recommendation engineer",
        "backend engineer",  # if combined with ML skills
    ],

    # ── Titles that are explicitly not a fit ──
    "negative_titles": [
        "marketing manager", "hr manager", "accountant",
        "sales executive", "content writer", "graphic designer",
        "mechanical engineer", "civil engineer", "customer support",
        "operations manager", "project manager",
    ],

    # ── Companies the JD explicitly says are bad fit ──
    "consulting_companies": [
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra",
    ],

    # ── Location preferences ──
    "preferred_locations": [
        "noida", "pune", "hyderabad", "mumbai", "delhi", "ncr",
        "gurgaon", "gurugram", "bengaluru", "bangalore",
    ],
    "preferred_countries": ["india"],

    # ── Notice period preference ──
    "max_notice_days": 30,  # Prefer sub-30 day; up to 30 buyout

    # ── Work mode ──
    "preferred_work_modes": ["hybrid", "flexible", "onsite"],
}

# =============================================================================
# Honeypot Detection Thresholds
# =============================================================================
HONEYPOT_MAX_SKILLS_WITH_ZERO_DURATION = 3
HONEYPOT_MAX_EXPERT_SKILLS = 8
HONEYPOT_MIN_CAREER_CONSISTENCY = 0.3  # Company founding vs experience mismatch

# =============================================================================
# Output
# =============================================================================
TOP_N = 100
