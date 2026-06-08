# AI Candidate Ranker Walkthrough

We have successfully built and verified the high-performance, CPU-only candidate ranking engine for the India Runs challenge.

## Changes Made
- **Performance Re-Architecting**: Transitioned from a slow end-to-end embedding generation strategy to an elegant **Two-Stage Multi-stage Retrieval pipeline** (L1 Filter + L2 Ranker).
  - **L1 Filter**: Performs O(1) exact-match skill lookup, fast substring-based job title screening, and platform availability checks on all 100,000 candidates (completing in under 3 seconds).
  - **L2 Ranker**: Focuses heavy dense-embedding representation (`sentence-transformers/all-MiniLM-L6-v2`) and fuzzy-logic matching (`rapidfuzz`) *only* on the top 5,000 screened candidates.
- **Honeypot Detection**: Implemented deterministic detection of synthetic impossible candidate records:
  - expert skill with zero months duration (expert in 3+ skills with 0 months experience).
  - job timelines with date mismatch (e.g. claiming 166 months of experience in a 24-month calendar duration).
  - Successfully identified and filtered **40 honeypot trap candidates**.
- **Streamlit Interface**: Completely updated `app/streamlit_app.py` to support the two-stage pipeline, providing team members with interactive configuration options and deep candidate profile breakdown charts.

## Performance Metrics
- **Dataset Size**: 100,000 candidates.
- **Ranking Step Execution Time**: **168.5 seconds** (~2.8 minutes), well under the strict 5-minute CPU constraint!
- **Honeypots Blocked**: 40.
- **Validation**: Confirmed to pass the challenge's official validator (`validate_submission.py`).

## Output Verification
The generated submission file `data/output/submission.csv` contains:
- Exactly 100 candidate rows.
- Ranked properly by descending scores.
- Comprehensive non-hallucinated reasoning fields.

For example, the top-ranked candidate:
- **ID**: `CAND_0071974` (Score: `0.7902`)
- **Reasoning**: `Senior AI Engineer at Netflix (Media); 7.8 yrs exp; 13 relevant AI/ML skills (LoRA, Learning to Rank, Weaviate, Pinecone); response rate 76%; 3 product-co roles; actively looking`
