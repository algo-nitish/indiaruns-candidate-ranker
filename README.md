# 🎯 AI Candidate Ranking System — INDIA RUNS Challenge

> Build an AI system that ranks candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     INPUT LAYER                          │
│   Job Description (JD)  +  Candidate Profiles Dataset    │
└──────────┬──────────────────────────┬────────────────────┘
           │                          │
           ▼                          ▼
┌────────────────────┐   ┌───────────────────────────────┐
│  JD Parser         │   │  Candidate Profile Parser     │
│  (LLM-powered)     │   │  (Auto column mapping)        │
└────────┬───────────┘   └──────────────┬────────────────┘
         │                              │
         ▼                              ▼
┌────────────────────┐   ┌───────────────────────────────┐
│  JD Embedding      │   │  Candidate Embeddings         │
│  (sentence-        │   │  (sentence-transformers)      │
│   transformers)    │   │                               │
└────────┬───────────┘   └──────────────┬────────────────┘
         └──────────┬───────────────────┘
                    ▼
         ┌──────────────────────┐
         │   HYBRID SCORING     │
         │                      │
         │  1. Semantic (30%)   │  ← Cosine similarity
         │  2. Skills (25%)     │  ← Fuzzy match + synonyms
         │  3. Experience (20%) │  ← Years + domain + level
         │  4. Behavioral (10%) │  ← Platform activity
         │  5. LLM Rerank (15%) │  ← GPT/Gemini judgment
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │   RANKED OUTPUT      │
         │   CSV + Rationale    │
         └──────────────────────┘
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/india-runs-ai-ranker.git
cd india-runs-ai-ranker

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, Gemini, or use Ollama for local)
```

### 3. Add Dataset

Place the challenge dataset files in `data/raw/`:
```
data/raw/
├── job_description.csv    # or .xlsx, .json
└── candidates.csv         # or .xlsx, .json
```

### 4. Run the Pipeline

```bash
# Explore available datasets
python -m src.main --explore

# Run full ranking pipeline
python -m src.main --jd data/raw/job_description.csv --candidates data/raw/candidates.csv

# Run without LLM (faster, no API key needed)
python -m src.main --jd data/raw/jd.csv --candidates data/raw/candidates.csv --no-llm

# Custom top-N and output format
python -m src.main --top-n 30 --output-format csv xlsx
```

### 5. Launch Demo UI

```bash
streamlit run app/streamlit_app.py
```

## 🧠 How It Works

### 1. Understanding the JD (Not Just Keywords)
The system uses an LLM to deeply parse job descriptions, extracting:
- **Must-have vs. nice-to-have skills** (with different weights)
- **Role level and experience expectations**
- **Industry context and soft skills**

### 2. Multi-Signal Scoring
Each candidate is evaluated across 5 dimensions:

| Signal | Weight | How It Works |
|--------|--------|-------------|
| **Semantic Similarity** | 30% | Embedding cosine similarity captures meaning, not just words |
| **Skills Match** | 25% | Fuzzy matching with 100+ synonym mappings |
| **Experience Relevance** | 20% | Years + role level + domain overlap |
| **Behavioral Signals** | 10% | Profile completeness, platform activity |
| **LLM Re-Ranking** | 15% | GPT/Gemini evaluates top candidates like a recruiter |

### 3. LLM Re-Ranking (The Secret Sauce)
The top 20 candidates are sent to an LLM with the JD context. The LLM evaluates:
- Career trajectory alignment
- Genuine vs. surface-level skill match
- Transferable experience
- Red flags (job hopping, overqualification)

Each candidate gets a **recommendation** (strong_yes / yes / maybe / no) with a human-readable **rationale**.

## 📁 Project Structure

```
IndiaRuns/
├── config/settings.py          # Central configuration
├── src/
│   ├── main.py                 # Pipeline orchestrator (CLI)
│   ├── parsers/
│   │   ├── jd_parser.py        # LLM-powered JD understanding
│   │   └── candidate_parser.py # Profile structuring
│   ├── embeddings/
│   │   ├── embedding_engine.py # Sentence-transformer embeddings
│   │   └── vector_store.py     # FAISS vector index
│   ├── scoring/
│   │   ├── semantic_scorer.py  # Embedding similarity
│   │   ├── skills_scorer.py    # Fuzzy skill matching
│   │   ├── experience_scorer.py# Experience relevance
│   │   ├── behavioral_scorer.py# Activity signals
│   │   └── hybrid_ranker.py    # Multi-signal fusion
│   ├── llm/
│   │   ├── llm_client.py       # OpenAI/Gemini/Ollama client
│   │   └── prompts.py          # Prompt templates
│   └── utils/
│       ├── data_loader.py      # Flexible data ingestion
│       └── export.py           # Output formatting
├── app/
│   └── streamlit_app.py        # Interactive demo UI
├── data/
│   ├── raw/                    # Input datasets
│   ├── processed/              # Intermediate data
│   └── output/                 # Ranked results
└── tests/                      # Test suite
```

## 🛠️ Tech Stack

- **Python 3.11+** — Core language
- **sentence-transformers** — Dense vector embeddings
- **FAISS** — Fast similarity search
- **OpenAI / Gemini / Ollama** — LLM for deep understanding
- **Pydantic** — Data validation
- **Pandas** — Data processing
- **Streamlit + Plotly** — Interactive demo UI
- **RapidFuzz** — Fuzzy string matching
- **Rich** — Beautiful terminal output

## 📊 Output Format

The ranked output CSV contains:

| Column | Description |
|--------|-------------|
| `rank` | Final ranking position |
| `candidate_id` | Unique candidate identifier |
| `name` | Candidate name |
| `composite_score` | Final hybrid score (0-1) |
| `semantic_score` | Embedding similarity score |
| `skills_score` | Weighted skills match score |
| `experience_score` | Experience relevance score |
| `behavioral_score` | Platform activity score |
| `llm_score` | LLM re-ranking score |
| `recommendation` | strong_yes / yes / maybe / no |
| `rationale` | LLM-generated fit explanation |

## 📄 License

MIT License — Built for the [INDIA RUNS](https://hack2skill.com/india-runs) Data & AI Challenge.
