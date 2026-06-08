# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Export Utilities — Generate ranked output files
# =============================================================================

import pandas as pd
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()


def export_ranked_candidates(
    ranked_df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "ranked_candidates",
    formats: list[str] = None,
) -> list[Path]:
    """
    Export ranked candidates to file(s).

    Args:
        ranked_df: DataFrame with ranked candidates (must have 'rank' column)
        output_dir: Directory to save output files
        filename: Base filename (without extension)
        formats: List of formats to export ("csv", "xlsx", "json")

    Returns:
        List of paths to created files
    """
    if formats is None:
        formats = ["csv"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_files = []

    for fmt in formats:
        output_path = output_dir / f"{filename}_{timestamp}.{fmt}"

        if fmt == "csv":
            ranked_df.to_csv(output_path, index=False, encoding="utf-8")
        elif fmt == "xlsx":
            ranked_df.to_excel(output_path, index=False, engine="openpyxl")
        elif fmt == "json":
            ranked_df.to_json(output_path, orient="records", indent=2)
        else:
            console.print(f"[yellow]⚠ Unsupported format: {fmt}[/yellow]")
            continue

        created_files.append(output_path)
        size_kb = output_path.stat().st_size / 1024
        console.print(f"[green]✓ Exported: {output_path.name} ({size_kb:.1f} KB)[/green]")

    return created_files


def format_ranked_output(
    candidates: list[dict],
    include_rationale: bool = True,
) -> pd.DataFrame:
    """
    Format ranked candidates into the standard output DataFrame.

    Expected input format (each candidate dict):
    {
        "candidate_id": str,
        "rank": int,
        "composite_score": float,
        "semantic_score": float,
        "skills_score": float,
        "experience_score": float,
        "behavioral_score": float,
        "llm_score": float,
        "rationale": str,       # Optional
        "name": str,            # Optional
        "top_skills": list,     # Optional
    }
    """
    df = pd.DataFrame(candidates)

    # Ensure rank column exists and is sorted
    if "rank" not in df.columns:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    else:
        df = df.sort_values("rank").reset_index(drop=True)

    # Round score columns
    score_cols = [c for c in df.columns if c.endswith("_score")]
    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Format top_skills as comma-separated string if present
    if "top_skills" in df.columns:
        df["top_skills"] = df["top_skills"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )

    # Remove rationale if not requested
    if not include_rationale and "rationale" in df.columns:
        df = df.drop(columns=["rationale"])

    # Reorder columns: rank first, then ID, scores, then extras
    priority_cols = ["rank", "candidate_id", "name", "composite_score"]
    other_cols = [c for c in df.columns if c not in priority_cols]
    ordered_cols = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[ordered_cols]

    return df
