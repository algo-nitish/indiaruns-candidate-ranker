# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Streamlit Demo UI — Interactive candidate ranking interface (Aligned with rank.py)
# =============================================================================
"""
Launch with:  streamlit run app/streamlit_app.py
"""

import os
os.environ["HF_HOME"] = r"d:\IndiaRuns\.cache\huggingface"

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

from config.settings import JD_REQUIREMENTS, SCORING_WEIGHTS
from src.scoring.candidate_scorer import CandidateScorer
from src.scoring.honeypot_detector import HoneypotDetector

# Page configuration
st.set_page_config(
    page_title="AI Candidate Ranker — INDIA RUNS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0b091a 0%, #151130 50%, #0d0c1d 100%);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #8f9bb3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00f2fe;
    }
    
    .metric-label {
        color: #8f9bb3;
        font-size: 0.9rem;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


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
    for job in c.get("career_history", [])[:3]:
        parts.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')[:200]}")
    skill_names = [s["name"] for s in c.get("skills", [])]
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names)}")
    return " | ".join(p for p in parts if p)


def build_jd_text() -> str:
    """Build JD text for embedding."""
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
    
    ai_keywords = {"nlp", "machine learning", "deep learning", "pytorch", "tensorflow",
                   "embeddings", "transformers", "bert", "gpt", "llm", "fine-tuning",
                   "rag", "vector", "faiss", "information retrieval", "search",
                   "recommendation", "ranking", "python", "scikit-learn", "xgboost",
                   "huggingface", "sentence-transformers", "lora", "qlora", "mlops",
                   "ml engineering", "data science"}
    relevant_skills = [s for s in skills if s.lower() in ai_keywords or
                       any(kw in s.lower() for kw in ["ml", "ai", "learn", "nlp", "embed",
                                                       "vector", "search", "rank", "retriev"])]
    
    parts = [
        f"{title} at {company} ({industry})",
        f"{yoe:.1f} yrs exp",
        f"{len(relevant_skills)} relevant skills" if relevant_skills else "0 relevant skills",
        f"response rate {resp_rate:.0%}"
    ]
    return "; ".join(parts)


def main():
    st.markdown('<h1 class="hero-title">🎯 AI Candidate Ranker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">INDIA RUNS Challenge — Interactive Dashboard</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Candidate Source
        st.subheader("👥 Candidates Data")
        data_source = st.radio("Load candidates from:", ["Sample Dataset (50 rows)", "Upload custom JSONL file"])
        
        candidates = []
        if data_source == "Sample Dataset (50 rows)":
            sample_path = PROJECT_ROOT / "required_datasets" / "India_runs_data_and_ai_challenge" / "sample_candidates.json"
            if sample_path.exists():
                with open(sample_path, "r", encoding="utf-8") as f:
                    candidates = json.load(f)
                st.success("Loaded 50 sample candidates.")
            else:
                st.error("Sample candidate file not found!")
        else:
            uploaded_file = st.file_uploader("Upload candidates.jsonl", type=["jsonl", "json"])
            if uploaded_file:
                for line in uploaded_file:
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))
                st.success(f"Loaded {len(candidates)} candidates.")

        st.subheader("🎛️ Scoring Weights")
        w_semantic = st.slider("Semantic (Embeddings) Weight", 0.0, 1.0, SCORING_WEIGHTS["semantic"], 0.05)
        w_skills = st.slider("Skills Match Weight", 0.0, 1.0, SCORING_WEIGHTS["skills_match"], 0.05)
        w_career = st.slider("Career Fit Weight", 0.0, 1.0, SCORING_WEIGHTS["career_fit"], 0.05)
        w_behavioral = st.slider("Behavioral Signals Weight", 0.0, 1.0, SCORING_WEIGHTS["behavioral"], 0.05)
        w_anti = st.slider("Anti-Patterns Weight", 0.0, 1.0, SCORING_WEIGHTS["anti_patterns"], 0.05)

        total_weight = w_semantic + w_skills + w_career + w_behavioral + w_anti
        if abs(total_weight - 1.0) > 0.001:
            st.warning(f"Weights sum to {total_weight:.2f}. They will be normalized.")

        st.subheader("Output Size")
        top_n = st.slider("Show Top Candidates", 5, 100, 20)

        run_ranking = st.button("🚀 Run Ranking", type="primary", use_container_width=True)

    # Main area
    if run_ranking and len(candidates) > 0:
        # Normalize weights
        norm_weights = {
            "semantic": w_semantic / total_weight,
            "skills_match": w_skills / total_weight,
            "career_fit": w_career / total_weight,
            "behavioral": w_behavioral / total_weight,
            "anti_patterns": w_anti / total_weight,
        }

        with st.spinner("⚡ Processing candidate profiles..."):
            scorer = CandidateScorer()
            detector = HoneypotDetector()
            
            # Step 1: Filter and score
            prelim_results = []
            honeypots_count = 0
            
            for c in candidates:
                if detector.is_honeypot(c):
                    honeypots_count += 1
                    continue
                
                scores = scorer.fast_score(c)
                prelim_score = (
                    scores["skills_match"] * norm_weights["skills_match"]
                    + scores["career_fit"] * norm_weights["career_fit"]
                    + scores["behavioral"] * norm_weights["behavioral"]
                    + scores["anti_patterns"] * norm_weights["anti_patterns"]
                )
                prelim_results.append({
                    "candidate_id": c["candidate_id"],
                    "prelim_score": prelim_score,
                    "scores": scores,
                    "candidate": c,
                })
            
            # Take up to 1000 for encoding in dashboard
            prelim_results.sort(key=lambda x: x["prelim_score"], reverse=True)
            subset = prelim_results[:1000]
            
            # Step 2: Semantic search
            model = SentenceTransformer("all-MiniLM-L6-v2")
            subset_texts = [build_candidate_text(r["candidate"]) for r in subset]
            jd_text = build_jd_text()
            
            candidate_embs = model.encode(subset_texts, batch_size=128, normalize_embeddings=True)
            jd_emb = model.encode([jd_text], normalize_embeddings=True)
            semantic_scores = np.dot(candidate_embs, jd_emb.T).flatten()
            
            # Step 3: Deep scoring
            results = []
            for idx, r in enumerate(subset):
                c = r["candidate"]
                scores = scorer.score(c)
                scores["semantic"] = float(semantic_scores[idx])
                
                composite = (
                    scores["semantic"] * norm_weights["semantic"]
                    + scores["skills_match"] * norm_weights["skills_match"]
                    + scores["career_fit"] * norm_weights["career_fit"]
                    + scores["behavioral"] * norm_weights["behavioral"]
                    + scores["anti_patterns"] * norm_weights["anti_patterns"]
                )
                
                results.append({
                    "candidate_id": r["candidate_id"],
                    "name": c.get("profile", {}).get("anonymized_name", "Anonymized"),
                    "title": c.get("profile", {}).get("current_title", "Unknown"),
                    "company": c.get("profile", {}).get("current_company", "Unknown"),
                    "composite_score": composite,
                    "scores": scores,
                    "candidate": c,
                })
            
            results.sort(key=lambda x: x["composite_score"], reverse=True)
            top_ranked = results[:top_n]
            
            # Store in session state
            st.session_state["ranked_results"] = top_ranked
            st.session_state["honeypots_count"] = honeypots_count
            st.session_state["total_candidates"] = len(candidates)

    if "ranked_results" in st.session_state:
        ranked_results = st.session_state["ranked_results"]
        honeypots_count = st.session_state["honeypots_count"]
        total_candidates = st.session_state["total_candidates"]
        
        # Display Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_candidates}</div>
                <div class="metric-label">Total Pool Size</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{honeypots_count}</div>
                <div class="metric-label">Honeypots Blocked</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ranked_results[0]['composite_score']:.4f}</div>
                <div class="metric-label">Highest Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📊 Rankings Table", "📈 Score Breakdown", "🔍 Candidate Deep-Dive"])
        
        with tab1:
            rows = []
            for rank, r in enumerate(ranked_results, start=1):
                reasoning = generate_reasoning(r["candidate"], r["scores"])
                rows.append({
                    "Rank": rank,
                    "Candidate ID": r["candidate_id"],
                    "Name": r["name"],
                    "Current Title": r["title"],
                    "Current Company": r["company"],
                    "Composite Score": round(r["composite_score"], 4),
                    "Reasoning": reasoning,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        with tab2:
            # Radar chart of top 5
            top5 = ranked_results[:5]
            categories = ["Semantic", "Skills Match", "Career Fit", "Behavioral", "Anti-Patterns"]
            
            fig = go.Figure()
            for rank, r in enumerate(top5, start=1):
                s = r["scores"]
                fig.add_trace(go.Scatterpolar(
                    r=[s["semantic"], s["skills_match"], s["career_fit"], s["behavioral"], s["anti_patterns"]],
                    theta=categories,
                    fill='toself',
                    name=f"#{rank} {r['name']}"
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    bgcolor="rgba(0,0,0,0)"
                ),
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            selected_idx = st.selectbox(
                "Select candidate to inspect:",
                range(len(ranked_results)),
                format_func=lambda x: f"#{x+1}: {ranked_results[x]['name']} ({ranked_results[x]['candidate_id']})"
            )
            
            r = ranked_results[selected_idx]
            c = r["candidate"]
            p = c.get("profile", {})
            
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.subheader(f"Profile: {r['name']}")
                st.markdown(f"**Headline:** {p.get('headline')}")
                st.markdown(f"**Current Title:** {p.get('current_title')} at {p.get('current_company')}")
                st.markdown(f"**Stated Experience:** {p.get('years_of_experience')} years")
                st.markdown(f"**Summary:** {p.get('summary')}")
                
                st.markdown("### 🛠️ Skills")
                skills_df = pd.DataFrame(c.get("skills", []))
                if not skills_df.empty:
                    st.dataframe(skills_df[["name", "proficiency", "duration_months", "endorsements"]], use_container_width=True, hide_index=True)
                    
                st.markdown("### 💼 Career History")
                for job in c.get("career_history", []):
                    st.markdown(f"**{job.get('title')}** at *{job.get('company')}* ({job.get('start_date')} to {job.get('end_date') or 'Present'})")
                    st.markdown(f"_{job.get('description')}_")
                    st.divider()
                    
            with col_r:
                # Gauge/Radar for the specific candidate
                fig = go.Figure(go.Scatterpolar(
                    r=[r["scores"]["semantic"], r["scores"]["skills_match"], r["scores"]["career_fit"], r["scores"]["behavioral"], r["scores"]["anti_patterns"]],
                    theta=categories,
                    fill='toself',
                    line_color="#00f2fe"
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="rgba(0,0,0,0)"),
                    showlegend=False,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Signals check
                st.markdown("### 📡 Platform Signals")
                sig = c.get("redrob_signals", {})
                st.write(f"**Response Rate:** {sig.get('recruiter_response_rate', 0):.0%}")
                st.write(f"**Avg Response Time:** {sig.get('avg_response_time_hours', 0)} hours")
                st.write(f"**GitHub Activity Score:** {sig.get('github_activity_score', 0)}")
                st.write(f"**Open to Work:** {sig.get('open_to_work_flag', False)}")
                st.write(f"**Notice Period:** {sig.get('notice_period_days', 90)} days")


if __name__ == "__main__":
    main()
