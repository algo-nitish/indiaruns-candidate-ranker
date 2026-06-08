# =============================================================================
# INDIA RUNS — Candidate Scorer
# Multi-signal scoring aligned with the actual JD requirements
# =============================================================================

from rapidfuzz import fuzz
from config.settings import JD_REQUIREMENTS

# Pre-compute lowercase sets for fast lookup
_MUST_HAVE = {s.lower() for s in JD_REQUIREMENTS["must_have_skills"]}
_NICE_TO_HAVE = {s.lower() for s in JD_REQUIREMENTS["nice_to_have_skills"]}
_POS_TITLES = {t.lower() for t in JD_REQUIREMENTS["positive_titles"]}
_NEG_TITLES = {t.lower() for t in JD_REQUIREMENTS["negative_titles"]}
_CONSULTING = {c.lower() for c in JD_REQUIREMENTS["consulting_companies"]}
_PREF_LOCS = {l.lower() for l in JD_REQUIREMENTS["preferred_locations"]}
_PREF_COUNTRIES = {c.lower() for c in JD_REQUIREMENTS["preferred_countries"]}


class CandidateScorer:
    """Score candidates against the hardcoded JD requirements. No API calls."""

    def fast_score(self, candidate: dict) -> dict:
        """Lightweight scoring method that runs in microseconds per candidate. Avoids fuzz/datetime parsing."""
        profile = candidate.get("profile", {})
        skills_raw = candidate.get("skills", [])
        candidate_skills = {s["name"].lower() for s in skills_raw}

        # 1. Skills (must-have exact intersections)
        must_hits = len(candidate_skills.intersection(_MUST_HAVE))
        must_score = must_hits / max(len(_MUST_HAVE), 1)

        # 2. Career fit title + experience
        title = profile.get("current_title", "").lower()
        title_score = 0.3
        # Fast substring check instead of fuzz
        if any(pos in title for pos in _POS_TITLES):
            title_score = 1.0
        if any(neg in title for neg in _NEG_TITLES):
            title_score = 0.05

        yoe = profile.get("years_of_experience", 0)
        min_y, max_y = JD_REQUIREMENTS["experience_range"]
        if min_y <= yoe <= max_y:
            exp_score = 1.0
        else:
            exp_score = 0.4

        career_fit = title_score * 0.6 + exp_score * 0.4

        # 3. Behavioral signals (direct lookup)
        sig = candidate.get("redrob_signals", {})
        resp_rate = sig.get("recruiter_response_rate", 0)
        behavioral = resp_rate

        # 4. Anti-patterns (fast checks)
        anti_patterns = 1.0
        if title_score <= 0.05:
            anti_patterns = 0.1

        return {
            "skills_match": must_score,
            "career_fit": career_fit,
            "behavioral": behavioral,
            "anti_patterns": anti_patterns,
        }

    def score(self, candidate: dict) -> dict:
        """Return dict of sub-scores, each in [0, 1]."""
        return {
            "skills_match": self._score_skills(candidate),
            "career_fit": self._score_career(candidate),
            "behavioral": self._score_behavioral(candidate),
            "anti_patterns": self._score_anti_patterns(candidate),
        }

    # ── Skills Match ──────────────────────────────────────────────────────
    def _score_skills(self, c: dict) -> float:
        skills_raw = c.get("skills", [])
        candidate_skills = {s["name"].lower() for s in skills_raw}
        candidate_skills_with_prof = {s["name"].lower(): s for s in skills_raw}

        # Must-have skill match (fuzzy)
        must_hits = 0
        for req_skill in _MUST_HAVE:
            if self._skill_matches(req_skill, candidate_skills):
                must_hits += 1
        must_score = must_hits / max(len(_MUST_HAVE), 1)

        # Nice-to-have skill match
        nice_hits = 0
        for req_skill in _NICE_TO_HAVE:
            if self._skill_matches(req_skill, candidate_skills):
                nice_hits += 1
        nice_score = nice_hits / max(len(_NICE_TO_HAVE), 1)

        # Skill depth: prefer advanced/expert with endorsements and duration
        depth_score = 0
        relevant_count = 0
        for s in skills_raw:
            name = s["name"].lower()
            if self._skill_matches_any(name, _MUST_HAVE | _NICE_TO_HAVE):
                relevant_count += 1
                prof_score = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.5, "beginner": 0.2}.get(
                    s.get("proficiency", "beginner"), 0.2)
                endorse_score = min(s.get("endorsements", 0) / 30.0, 1.0)
                duration_score = min(s.get("duration_months", 0) / 36.0, 1.0)
                depth_score += (prof_score * 0.4 + endorse_score * 0.3 + duration_score * 0.3)

        depth_avg = depth_score / max(relevant_count, 1)

        # Skill assessment scores from Redrob
        assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
        assess_score = 0
        if assessments:
            relevant_assess = [v for k, v in assessments.items()
                               if self._skill_matches_any(k.lower(), _MUST_HAVE | _NICE_TO_HAVE)]
            if relevant_assess:
                assess_score = sum(relevant_assess) / (len(relevant_assess) * 100)

        return (must_score * 0.40 + nice_score * 0.15 + depth_avg * 0.30 + assess_score * 0.15)

    # ── Career Fit ────────────────────────────────────────────────────────
    def _score_career(self, c: dict) -> float:
        profile = c.get("profile", {})
        career = c.get("career_history", [])
        education = c.get("education", [])

        scores = []

        # 1. Title alignment (30%)
        title = profile.get("current_title", "").lower()
        title_score = 0.3  # neutral default
        for pos in _POS_TITLES:
            if fuzz.partial_ratio(pos, title) > 80:
                title_score = 1.0
                break
        for neg in _NEG_TITLES:
            if fuzz.partial_ratio(neg, title) > 85:
                title_score = 0.05  # near-zero for explicitly bad titles
                break
        scores.append(("title", title_score, 0.30))

        # 2. Experience years (20%)
        yoe = profile.get("years_of_experience", 0)
        min_y, max_y = JD_REQUIREMENTS["experience_range"]
        if min_y <= yoe <= max_y:
            exp_score = 1.0
        elif yoe < min_y:
            exp_score = max(0, 1.0 - (min_y - yoe) / 5)
        else:
            exp_score = max(0.3, 1.0 - (yoe - max_y) / 10)
        scores.append(("experience", exp_score, 0.20))

        # 3. Product company vs consulting (25%)
        product_score = 0.5
        has_consulting_only = True
        for job in career:
            company = job.get("company", "").lower()
            industry = job.get("industry", "").lower()
            is_consulting = any(cc in company for cc in _CONSULTING) or industry == "it services"
            if not is_consulting:
                has_consulting_only = False
                # Product company experience is a strong positive
                if job.get("company_size", "") in ["11-50", "51-200", "201-500", "501-1000"]:
                    product_score = max(product_score, 0.9)
                else:
                    product_score = max(product_score, 0.7)

        if has_consulting_only and len(career) > 0:
            product_score = 0.1  # JD explicitly says consulting-only is bad fit
        scores.append(("product_co", product_score, 0.25))

        # 4. Career description relevance (15%)
        desc_keywords = ["ranking", "retrieval", "search", "recommendation", "embedding",
                         "nlp", "ml", "machine learning", "ai", "data pipeline",
                         "production", "deployed", "a/b test", "model", "inference"]
        desc_hits = 0
        desc_total = 0
        for job in career:
            desc = job.get("description", "").lower()
            for kw in desc_keywords:
                if kw in desc:
                    desc_hits += 1
            desc_total += len(desc_keywords)
        desc_score = desc_hits / max(desc_total, 1)
        scores.append(("descriptions", min(desc_score * 3, 1.0), 0.15))

        # 5. Location match (10%)
        location = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        loc_score = 0.3
        if country in _PREF_COUNTRIES:
            loc_score = 0.6
            for loc in _PREF_LOCS:
                if loc in location:
                    loc_score = 1.0
                    break
        scores.append(("location", loc_score, 0.10))

        return sum(s * w for _, s, w in scores)

    # ── Behavioral Signals ────────────────────────────────────────────────
    def _score_behavioral(self, c: dict) -> float:
        sig = c.get("redrob_signals", {})

        components = []

        # Recruiter response rate (critical — JD emphasizes availability)
        resp_rate = sig.get("recruiter_response_rate", 0)
        components.append(("resp_rate", resp_rate, 0.25))

        # Response time (lower is better)
        resp_time = sig.get("avg_response_time_hours", 200)
        resp_time_score = max(0, 1.0 - resp_time / 200)
        components.append(("resp_time", resp_time_score, 0.10))

        # Profile completeness
        completeness = sig.get("profile_completeness_score", 0) / 100
        components.append(("completeness", completeness, 0.10))

        # Open to work
        otw = 1.0 if sig.get("open_to_work_flag", False) else 0.3
        components.append(("open_to_work", otw, 0.10))

        # Recent activity (last_active_date)
        last_active = sig.get("last_active_date", "2020-01-01")
        from datetime import datetime
        try:
            la_date = datetime.strptime(last_active, "%Y-%m-%d")
            days_ago = (datetime(2026, 6, 8) - la_date).days
            recency = max(0, 1.0 - days_ago / 365)
        except Exception:
            recency = 0.3
        components.append(("recency", recency, 0.15))

        # Notice period (prefer short)
        notice = sig.get("notice_period_days", 90)
        notice_score = 1.0 if notice <= 30 else max(0.2, 1.0 - (notice - 30) / 120)
        components.append(("notice", notice_score, 0.10))

        # GitHub activity
        github = sig.get("github_activity_score", -1)
        github_score = 0.3  # neutral if no github
        if github >= 0:
            github_score = min(github / 80, 1.0)
        components.append(("github", github_score, 0.10))

        # Interview + offer signals
        interview_rate = sig.get("interview_completion_rate", 0.5)
        offer_rate = sig.get("offer_acceptance_rate", -1)
        offer_score = 0.5 if offer_rate < 0 else offer_rate
        components.append(("interview", interview_rate, 0.05))
        components.append(("offer", offer_score, 0.05))

        return sum(s * w for _, s, w in components)

    # ── Anti-Pattern Detection ────────────────────────────────────────────
    def _score_anti_patterns(self, c: dict) -> float:
        """Higher score = fewer red flags = better candidate."""
        penalties = 0.0

        profile = c.get("profile", {})
        career = c.get("career_history", [])
        skills = c.get("skills", [])
        title = profile.get("current_title", "").lower()

        # Penalty: title is explicitly wrong for role
        for neg in _NEG_TITLES:
            if fuzz.partial_ratio(neg, title) > 85:
                penalties += 0.3
                break

        # Penalty: job hopping (many short stints)
        short_stints = sum(1 for j in career if j.get("duration_months", 0) < 18)
        if short_stints >= 3:
            penalties += 0.15

        # Penalty: consulting-only career
        consulting_only = all(
            any(cc in j.get("company", "").lower() for cc in _CONSULTING)
            or j.get("industry", "").lower() == "it services"
            for j in career
        ) if career else False
        if consulting_only:
            penalties += 0.2

        # Penalty: skills don't match career descriptions (keyword stuffer signal)
        ai_skill_count = sum(1 for s in skills if self._skill_matches_any(
            s["name"].lower(), _MUST_HAVE | _NICE_TO_HAVE))
        desc_text = " ".join(j.get("description", "").lower() for j in career)
        desc_has_ml = any(kw in desc_text for kw in ["ml", "machine learning", "ai", "model",
                                                       "embedding", "ranking", "retrieval", "nlp"])
        if ai_skill_count >= 5 and not desc_has_ml:
            penalties += 0.25  # High AI skills but no AI in career = keyword stuffer

        # Penalty: title mismatch with skills (marketing manager with 10 AI skills)
        if any(neg in title for neg in ["marketing", "hr ", "accountant", "sales",
                                         "graphic", "content writer", "civil", "mechanical"]):
            if ai_skill_count >= 6:
                penalties += 0.3  # Strong keyword stuffer signal

        return max(0, 1.0 - penalties)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _skill_matches(self, target: str, candidate_skills: set) -> bool:
        if target in candidate_skills:
            return True
        for cs in candidate_skills:
            if fuzz.partial_ratio(target, cs) > 85:
                return True
        return False

    def _skill_matches_any(self, skill: str, target_set: set) -> bool:
        if skill in target_set:
            return True
        for t in target_set:
            if fuzz.partial_ratio(skill, t) > 80:
                return True
        return False
