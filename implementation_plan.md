# INDIA RUNS — Revised Plan (Based on Actual Dataset)

## Critical Constraints
- **100K candidates** (JSONL, 487MB)
- **Output: exactly 100 rows** — `candidate_id,rank,score,reasoning`
- **CPU only, 16GB RAM, 5 min max** — NO LLM API calls during ranking
- **~80 honeypot candidates** with impossible profiles — must detect and exclude
- **Keyword stuffing is a trap** — JD explicitly warns against it
- **Behavioral signals** are critical multipliers

## Architecture: Pre-compute + Fast Rank

### Phase 1: Pre-computation (can take longer)
- Generate embeddings for all 100K candidates
- Parse JD into structured requirements
- Save embeddings/indexes to disk

### Phase 2: Ranking (must complete in 5 min on CPU)
- Load pre-computed embeddings
- Multi-signal scoring (no API calls)
- Honeypot detection
- Output top 100 CSV

## Entry point
```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
