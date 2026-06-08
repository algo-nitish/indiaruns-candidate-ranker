# 🎯 India Runs Submission Deck Content
This document contains the exact slide-by-slide copy-pasteable content for your Google Slides presentation based on the official template. 

---

## Slide 1: Title Slide
- **Title**: `INDIA.RUNS (Build what next India runs on)`
- **Subtitle**: `AI-Powered Candidate Ranker: Two-Stage Hybrid Engine`
- **Team Name**: `IndiaRuns AI Rankers`
- **Team Members**: `Nitin & Antigravity AI`
- **Problem Statement**:
  > Conventional candidate matching systems rely on static keyword filtering, which is highly vulnerable to keyword-stuffing exploits and unable to detect impossible/fraudulent profile details. We build a high-fidelity, CPU-constrained ranking system that screens out synthetic "honeypot" profiles and evaluates candidates across multi-dimensional criteria: semantic alignment, career trajectory, skill proficiency/duration, and real-time behavioral signals.

---

## Slide 2: Solution Overview
- **Proposed Solution**:
  - We designed a **Two-Stage Multi-stage Candidate Ranker** that handles 100K candidates under strict CPU compute limits.
  - **Stage 1 (L1 Filter)**: Ultra-fast rule-based screening to remove honeypots and select the top 5,000 candidates based on set-intersection skill matches.
  - **Stage 2 (L2 Ranker)**: Semantic embedding similarity (`SentenceTransformer`) + high-fidelity Candidate Scoring on the top 5,000 candidates.
- **Key Differentiators**:
  - **Zero LLM APIs at Runtime**: Runs fully offline, CPU-only, completing in 2.8 minutes for 100K profiles.
  - **Honeypot Resilience**: Fully filters out fraudulent profiles using impossibility validation checks.
  - **Behavioral Signal Fusion**: Boosts candidates based on platform activity, response rates, and notice periods.

---

## Slide 3: JD Understanding & Candidate Evaluation
- **Key JD Requirements Extracted**:
  - **Must-Have**: Python, Vector databases (FAISS, Pinecone, Weaviate, Milvus, Qdrant), Embeddings-based retrieval, ranking evaluation (NDCG, MRR).
  - **Nice-to-Have**: LLM fine-tuning (LoRA, QLoRA), Learning-to-Rank, distributed systems.
  - **Target Experience**: 5 to 9 years of experience.
  - **Target Background**: Product-company experience preferred (consulting-only penalized).
- **Candidate Evaluation Signals**:
  - **Semantic Fit (30%)**: Cosine similarity between candidate text (profile summary + career history) and the JD.
  - **Skill Match (25%)**: Exact matches for must-haves + nice-to-haves, with depth scoring based on proficiency level and duration.
  - **Career fit (20%)**: Title alignment, years of experience, and product-company background.
  - **Behavioral (15%)**: Recruiter response rate, activity recency, and short notice periods.
  - **Anti-patterns (10%)**: Job hopping, keyword stuffing, and title-skill mismatches.

---

## Slide 4: Ranking Methodology
- **Multi-Stage Retrieval & Scoring Pipeline**:
  1. **L1 Filter**:
     - Fast regex/string-split timeline check to detect honeypots.
     - Exact skill intersection (O(1) in C/Python set operations).
     - Lightweight score calculation to rank and filter the pool to the top 5,000 candidates in under 3 seconds.
  2. **L2 Detailed Ranking**:
     - Load `all-MiniLM-L6-v2` (sentence-transformer) to encode candidate profiles and compute cosine similarity against the JD text.
     - Run the detailed `CandidateScorer` (incorporating `rapidfuzz` for fuzzy matches and comprehensive signal calculations).
     - Compute the weighted composite score for final sorting.

---

## Slide 5: Explainability & Data Validation
- **Explainable AI (XAI)**:
  - Generates clear, non-hallucinated explanations for each candidate's rank based directly on their profile data (e.g., specific skills matched, years of experience, current employer).
- **Fraud & Honeypot Detection**:
  - **Timeline Impossibility Check**: Flags profiles where stated job duration exceeds calendar time elapsed (e.g., claiming 166 months of experience in a 24-month calendar duration).
  - **Fake Skill Check**: Flags profiles claiming "expert" proficiency in multiple skills but stating 0 months of experience.
  - **Anti-Keyword Stuffing**: Applies strict penalties to profiles with high AI skill counts but no matching career description.

---

## Slide 6: End-to-End Workflow
- **Step-by-Step Execution**:
  1. **Load Data**: Stream candidates from `candidates.jsonl`.
  2. **Fast Filter**: Honeypot detection + set-intersection scoring.
  3. **L1 Top Selection**: Sort and select the top 5,000 profiles.
  4. **Embedding Generation**: Extract embeddings using sentence-transformers.
  5. **Signal Fusion**: Calculate composite scores across all metrics.
  6. **Validator Check**: Export the top 100 rows to `submission.csv` and execute `validate_submission.py`.

---

## Slide 7: System Architecture
- *(Action: Insert the generated image `system_architecture.png` here)*

---

## Slide 8: Results & Performance
- **Runtime Performance**:
  - Entire 100K dataset processed, ranked, and validated in **168.5 seconds** (~2.8 minutes).
- **Honeypots Blocked**: **40 fraudulent profiles** detected and filtered.
- **Top Candidate**: `CAND_0071974` (Score: `0.7902`) — Senior AI Engineer at Netflix, 7.8 years exp, expert in LoRA, Weaviate, Pinecone, and Learning to Rank.
- **Validation**: 100% compliance with challenge specifications.

---

## Slide 9: Technologies Used
- **Python 3.12**: Core programming language.
- **Sentence-Transformers**: Open-source embedding model `all-MiniLM-L6-v2`.
- **RapidFuzz**: High-performance C++-backed fuzzy string matching library.
- **Pandas & NumPy**: Fast vector and data manipulation.
- **Pytest**: Core testing framework.
- **Streamlit**: Premium web demonstration UI.

---

## Slide 10: Submission Assets
- **GitHub Repository**: [https://github.com/nitin/indiaruns-candidate-ranker](https://github.com/nitin/indiaruns-candidate-ranker)
- **Hosted Sandbox Demo**: [https://indiaruns-candidate-ranker.streamlit.app/](https://indiaruns-candidate-ranker.streamlit.app/)
- **Output Submission**: `data/output/submission.csv`

---

## Slide 11: Thank You
- **Closing**:
  - NITIN — Nitin's Contact details / nitin@example.com
  - ANTIGRAVITY AI
