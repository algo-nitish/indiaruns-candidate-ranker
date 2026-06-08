# =============================================================================
# INDIA RUNS — Honeypot Detector
# Identifies ~80 candidates with subtly impossible profiles
# =============================================================================


class HoneypotDetector:
    """
    Detects honeypot candidates with impossible profiles:
    - Expert in 3+ skills with 0 duration months
    - Career timeline duration mismatch (e.g. 166 months starting in 2023)
    """

    def is_honeypot(self, candidate: dict) -> bool:
        """Returns True if the candidate is a honeypot."""
        if self._check_expert_zero_duration(candidate):
            return True
        if self._check_timeline_anomaly(candidate):
            return True
        return False

    def _check_expert_zero_duration(self, c: dict) -> bool:
        """Expert in 3+ skills with 0 duration months."""
        skills = c.get("skills", [])
        expert_zero = sum(1 for s in skills
                          if s.get("proficiency") == "expert"
                          and s.get("duration_months", 0) == 0)
        return expert_zero >= 3

    def _check_timeline_anomaly(self, c: dict) -> bool:
        """Timeline duration does not match start/end dates. Uses fast string splitting."""
        career = c.get("career_history", [])

        for job in career:
            start_str = job.get("start_date")
            end_str = job.get("end_date")
            duration_months = job.get("duration_months", 0)

            if not start_str or "-" not in start_str:
                continue

            try:
                # Fast split instead of datetime.strptime
                s_parts = start_str.split("-")
                start_year = int(s_parts[0])
                start_month = int(s_parts[1])

                if end_str and "-" in end_str:
                    e_parts = end_str.split("-")
                    end_year = int(e_parts[0])
                    end_month = int(e_parts[1])
                else:
                    end_year = 2026
                    end_month = 6

                # Calculate calendar months
                calendar_months = (end_year - start_year) * 12 + (end_month - start_month)

                if duration_months > calendar_months + 12 and duration_months > 24:
                    return True

            except Exception:
                pass

        return False
