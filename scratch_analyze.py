import json
from pathlib import Path

DATA_DIR = Path(r"d:\IndiaRuns\required_datasets\India_runs_data_and_ai_challenge")
candidates_file = DATA_DIR / "candidates.jsonl"

print("Scanning first 20,000 candidates for honeypots...")

count = 0
with open(candidates_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        
        # Check 1: Expert skills with 0 duration
        skills = c.get("skills", [])
        expert_zero = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
        
        # Check 2: Experience timeline anomaly
        career = c.get("career_history", [])
        timeline_anomaly = False
        for job in career:
            # e.g., duration is much longer than the company could have existed, or similar inconsistencies.
            # Wait, does the job list company founding year? No, the schema doesn't have company founding year!
            # Ah! Let's check the schema again. The schema has:
            # company, title, start_date, end_date, duration_months, is_current, industry, company_size, description.
            # Wait, how could a candidate have "8 years of experience at a company founded 3 years ago"?
            # Let's check if the description or company name or something has it. Or maybe duration_months is impossible compared to start_date and end_date!
            # E.g. start_date is 2023 and end_date is 2026, but duration_months is 96 (8 years)!
            # Let's check that!
            start_date = job.get("start_date")
            end_date = job.get("end_date") or "2026-06-08" # current date
            duration_months = job.get("duration_months", 0)
            
            if start_date and end_date:
                try:
                    from datetime import datetime
                    s_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    e_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    actual_months = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                    if abs(duration_months - actual_months) > 24 and duration_months > 0:
                        timeline_anomaly = True
                except Exception:
                    pass

        if expert_zero >= 3 or timeline_anomaly:
            p = c.get("profile", {})
            print(f"Honeypot candidate {c['candidate_id']}: {p.get('anonymized_name')}")
            print(f"  expert_zero: {expert_zero}, timeline_anomaly: {timeline_anomaly}")
            if timeline_anomaly:
                for job in career:
                    print(f"    - Job: {job.get('company')}, duration={job.get('duration_months')}, start={job.get('start_date')}, end={job.get('end_date')}")
            count += 1
            if count >= 10:
                break
