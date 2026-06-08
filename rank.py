#!/usr/bin/env python3
"""
INDIA RUNS — AI Candidate Ranking System
Main entry point: produces submission.csv from candidates.jsonl

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py  (uses default paths from config)
"""
import os
os.environ["HF_HOME"] = r"d:\IndiaRuns\.cache\huggingface"

import argparse
import json
import sys

# Reconfigure stdout/stderr to UTF-8 to handle unicode symbols in terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

from config.settings import (
    CANDIDATES_FILE, OUTPUT_DIR, TOP_N,
    JD_REQUIREMENTS, SCORING_WEIGHTS, PRECOMPUTED_DIR,
)
from src.scoring.candidate_scorer import CandidateScorer
from src.scoring.honeypot_detector import HoneypotDetector

console = Console()


def load_candidates(path: str | Path) -> list[dict]:
    """Load candidates from JSONL or JSON file."""
    path = Path(path)
    console.print(f"[cyan]📂 Loading candidates from {path.name}...[/cyan]")
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith('['):
            try:
                candidates = json.loads(content)
            except Exception as e:
                console.print(f"[red]Failed to load as JSON array: {e}. Trying JSONL format...[/red]")
                # fallback to line-by-line
                candidates = []
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    console.print(f"[green]✓ Loaded {len(candidates)} candidates[/green]")
    return candidates


def build_candidate_text(c: dict) -> str:
    """Build a rich text representation for embedding."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        f"Current: {p.get('current_title', '')} at {p.get('current_company', '')}",
        f"Industry: {p.get('current_industry', '')}",
        f"Experience: {p.get('years_of_experience', 0)} years",
    ]
    # Add career history descriptions
    for job in c.get("career_history", [])[:3]:
        parts.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')[:200]}")
    # Add skills
    skill_names = [s["name"] for s in c.get("skills", [])]
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names)}")
    return " | ".join(p for p in parts if p)


def build_jd_text() -> str:
    """Build JD text for embedding from hardcoded requirements."""
    jd = JD_REQUIREMENTS
    return (
        f"Senior AI/ML Engineer at HR-tech product company. "
        f"Building ranking, retrieval, and matching systems for recruiting platform. "
        f"Must have: production embeddings-based retrieval (sentence-transformers, BGE, E5), "
        f"vector databases (FAISS, Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch), "
        f"strong Python, ranking evaluation (NDCG, MRR, MAP, A/B testing). "
        f"Nice to have: LLM fine-tuning (LoRA, QLoRA), learning-to-rank, HR-tech experience, "
        f"distributed systems. 5-9 years experience. Product company background preferred. "
        f"Must write production code. NLP/IR background required."
    )


def generate_reasoning(c: dict, scores: dict) -> str:
    """Generate concise, specific reasoning for a candidate's ranking."""
    p = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    skills = [s["name"] for s in c.get("skills", [])]
    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    company = p.get("current_company", "Unknown")
    industry = p.get("current_industry", "Unknown")
    resp_rate = signals.get("recruiter_response_rate", 0)
    
    # Count relevant AI/ML skills
    ai_keywords = {"nlp", "machine learning", "deep learning", "pytorch", "tensorflow",
                   "embeddings", "transformers", "bert", "gpt", "llm", "fine-tuning",
                   "rag", "vector", "faiss", "information retrieval", "search",
                   "recommendation", "ranking", "python", "scikit-learn", "xgboost",
                   "huggingface", "sentence-transformers", "lora", "qlora", "mlops",
                   "ml engineering", "data science", "neural networks", "computer vision",
                   "spacy", "langchain", "pinecone", "weaviate", "elasticsearch",
                   "milvus", "qdrant", "opensearch", "spark", "airflow", "kubernetes",
                   "docker", "aws", "gcp", "azure", "sql", "nosql", "mongodb",
                   "image classification", "object detection", "speech recognition",
                   "statistical modeling", "feature engineering", "model deployment",
                   "a/b testing", "experiment tracking", "weights & biases", "wandb",
                   "bentoml", "kubeflow", "sagemaker"}
    relevant_skills = [s for s in skills if s.lower() in ai_keywords or
                       any(kw in s.lower() for kw in ["ml", "ai", "learn", "nlp", "embed",
                                                       "vector", "search", "rank", "retriev"])]
    
    parts = []
    parts.append(f"{title} at {company} ({industry})")
    parts.append(f"{yoe:.1f} yrs exp")
    if relevant_skills:
        parts.append(f"{len(relevant_skills)} relevant AI/ML skills ({', '.join(relevant_skills[:4])})")
    parts.append(f"response rate {resp_rate:.0%}")
    
    # Add career insight
    career = c.get("career_history", [])
    product_cos = sum(1 for j in career if j.get("company_size", "") not in ["10001+"] 
                      or j.get("industry", "").lower() not in ["it services"])
    if product_cos > 0:
        parts.append(f"{product_cos} product-co roles")
    
    # Behavioral note
    if signals.get("open_to_work_flag"):
        parts.append("actively looking")
    if signals.get("notice_period_days", 999) <= 30:
        parts.append(f"notice {signals['notice_period_days']}d")
    
    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="AI Candidate Ranking System")
    parser.add_argument("--candidates", type=str, default=str(CANDIDATES_FILE))
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "submission.csv"))
    parser.add_argument("--precompute", action="store_true", help="Run pre-computation (embeddings)")
    args = parser.parse_args()

    console.print("\n[bold cyan]═══ INDIA RUNS — AI Candidate Ranker ═══[/bold cyan]\n")
    start = time.time()

    # Load candidates
    candidates = load_candidates(args.candidates)

    # Initialize scorers
    scorer = CandidateScorer()
    honeypot_detector = HoneypotDetector()

    # ─── RANKING STEP (must complete in 5 min on CPU) ───
    console.print("\n[bold yellow]⚡ Running ranking step...[/bold yellow]\n")
    rank_start = time.time()

    # Step 1: Detect honeypots and run lightweight preliminary scoring on all candidates
    console.print("[cyan]🍯 Detecting honeypots and running preliminary filter...[/cyan]")
    honeypot_ids = set()
    prelim_results = []

    total_candidates = len(candidates)
    for idx, c in enumerate(candidates):
        if idx % 20000 == 0:
            console.print(f"   Processed {idx}/{total_candidates} candidates...")
        cid = c["candidate_id"]

        # Skip honeypots entirely
        if honeypot_detector.is_honeypot(c):
            honeypot_ids.add(cid)
            continue

        # Score other parameters (fast rule-based scoring)
        scores = scorer.fast_score(c)

        # Preliminary score based on skills, career fit, behavioral and anti-patterns
        # We weight them according to settings.py weights (excluding semantic)
        prelim_score = (
            scores["skills_match"] * SCORING_WEIGHTS["skills_match"]
            + scores["career_fit"] * SCORING_WEIGHTS["career_fit"]
            + scores["behavioral"] * SCORING_WEIGHTS["behavioral"]
            + scores["anti_patterns"] * SCORING_WEIGHTS["anti_patterns"]
        )

        prelim_results.append({
            "candidate_id": cid,
            "prelim_score": prelim_score,
            "scores": scores,
            "candidate": c,
        })

    console.print(f"[yellow]   Flagged {len(honeypot_ids)} honeypots[/yellow]")
    console.print(f"[green]✓ Filtered candidates down from {len(candidates)}[/green]")

    # Sort and take the top 5,000 candidates for dense semantic search
    prelim_results.sort(key=lambda x: x["prelim_score"], reverse=True)
    candidate_subset = prelim_results[:5000]

    console.print(f"[cyan]🔢 Running sentence-transformers semantic search on top 5,000 candidates...[/cyan]")
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

    model = SentenceTransformer(EMBEDDING_MODEL)
    subset_texts = [build_candidate_text(r["candidate"]) for r in candidate_subset]
    jd_text = build_jd_text()

    # Encode only the subset (takes about 10-15 seconds on CPU!)
    candidate_embs = model.encode(subset_texts, batch_size=EMBEDDING_BATCH_SIZE,
                                   show_progress_bar=True, normalize_embeddings=True)
    jd_emb = model.encode([jd_text], normalize_embeddings=True)

    # Compute semantic scores
    semantic_scores = np.dot(candidate_embs, jd_emb.T).flatten()

    # Step 2: Combine scores and compute final composite scores
    results = []
    for idx, r in enumerate(candidate_subset):
        # Compute the deep high-fidelity scores for the top 5000 subset
        scores = scorer.score(r["candidate"])
        scores["semantic"] = float(semantic_scores[idx])

        # Compute composite score including the semantic embedding score
        composite = (
            scores["semantic"] * SCORING_WEIGHTS["semantic"]
            + scores["skills_match"] * SCORING_WEIGHTS["skills_match"]
            + scores["career_fit"] * SCORING_WEIGHTS["career_fit"]
            + scores["behavioral"] * SCORING_WEIGHTS["behavioral"]
            + scores["anti_patterns"] * SCORING_WEIGHTS["anti_patterns"]
        )

        results.append({
            "candidate_id": r["candidate_id"],
            "composite_score": composite,
            "scores": scores,
            "candidate": r["candidate"],
        })

    # Step 3: Sort and take top 100
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_100 = results[:TOP_N]

    # Step 4: Generate output CSV
    rows = []
    for rank, r in enumerate(top_100, start=1):
        reasoning = generate_reasoning(r["candidate"], r["scores"])
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": round(r["composite_score"], 4),
            "reasoning": reasoning,
        })

    df = pd.DataFrame(rows)

    # Ensure scores are non-increasing (required by validator)
    for i in range(1, len(df)):
        if df.loc[i, "score"] > df.loc[i-1, "score"]:
            df.loc[i, "score"] = df.loc[i-1, "score"]

    # Tiebreak: same score → candidate_id ascending
    df = df.sort_values(["score", "candidate_id"], ascending=[False, True])
    df["rank"] = range(1, len(df) + 1)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    rank_time = time.time() - rank_start
    total_time = time.time() - start

    console.print(f"\n[bold green]✅ Done![/bold green]")
    console.print(f"   Ranking step: {rank_time:.1f}s")
    console.print(f"   Total time: {total_time:.1f}s")
    console.print(f"   Honeypots filtered: {len(honeypot_ids)}")
    console.print(f"   Output: {out_path}")
    console.print(f"   Top candidate: {top_100[0]['candidate_id']} "
                   f"(score={top_100[0]['composite_score']:.4f})")

    # Validate
    console.print("\n[cyan]🔍 Running validator...[/cyan]")
    sys.path.insert(0, str(Path(args.candidates).parent))
    try:
        from required_datasets.India_runs_data_and_ai_challenge.validate_submission import validate_submission
        errors = validate_submission(str(out_path))
        if errors:
            console.print(f"[red]Validation failed ({len(errors)} issues):[/red]")
            for e in errors:
                console.print(f"  [red]• {e}[/red]")
        else:
            console.print("[bold green]✓ Submission is valid![/bold green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not run validator: {e}[/yellow]")


if __name__ == "__main__":
    main()
