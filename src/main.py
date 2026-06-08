# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Main Pipeline Orchestrator
# =============================================================================
"""
Entry point for the AI Candidate Ranking System.

Usage:
    python -m src.main --jd <job_description_file> --candidates <candidates_file>
    python -m src.main --jd-text "We are looking for a Senior Python Developer..."
    python -m src.main --explore  (explore available datasets)
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_RAW_DIR, DATA_OUTPUT_DIR, OUTPUT_TOP_N,
    INCLUDE_RATIONALE, LLM_PROVIDER,
)
from src.utils.data_loader import auto_load, discover_datasets, profile_dataframe
from src.utils.export import export_ranked_candidates, format_ranked_output
from src.parsers.jd_parser import JDParser
from src.parsers.candidate_parser import CandidateParser
from src.embeddings.embedding_engine import EmbeddingEngine
from src.scoring.hybrid_ranker import HybridRanker
from src.llm.llm_client import create_llm_client_from_config

console = Console()


def print_banner():
    """Print the startup banner."""
    banner = """
[bold cyan]
 ██╗███╗   ██╗██████╗ ██╗ █████╗     ██████╗ ██╗   ██╗███╗   ██╗███████╗
 ██║████╗  ██║██╔══██╗██║██╔══██╗    ██╔══██╗██║   ██║████╗  ██║██╔════╝
 ██║██╔██╗ ██║██║  ██║██║███████║    ██████╔╝██║   ██║██╔██╗ ██║███████╗
 ██║██║╚██╗██║██║  ██║██║██╔══██║    ██╔══██╗██║   ██║██║╚██╗██║╚════██║
 ██║██║ ╚████║██████╔╝██║██║  ██║    ██║  ██║╚██████╔╝██║ ╚████║███████║
 ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
[/bold cyan]
[bold white]     AI-Powered Candidate Ranking System[/bold white]
[dim]     Semantic Search · LLM Re-Ranking · Hybrid Scoring[/dim]
"""
    console.print(banner)


def explore_data():
    """Explore available datasets."""
    console.print("[bold]🔍 Exploring available datasets...\n[/bold]")
    files = discover_datasets(DATA_RAW_DIR)

    for f in files:
        try:
            df = auto_load(f)
            profile_dataframe(df, f.stem)
        except Exception as e:
            console.print(f"[red]✗ Error loading {f.name}: {e}[/red]")


def run_pipeline(
    jd_source: str,
    candidates_source: str,
    top_n: int = OUTPUT_TOP_N,
    use_llm: bool = True,
    output_formats: list[str] = None,
):
    """
    Run the full ranking pipeline.

    Args:
        jd_source: Path to JD file or raw JD text
        candidates_source: Path to candidates dataset file
        top_n: Number of top candidates to output
        use_llm: Whether to use LLM for re-ranking
        output_formats: List of output formats (csv, xlsx, json)
    """
    output_formats = output_formats or ["csv"]

    console.print(Panel(
        f"[bold]JD Source:[/bold] {jd_source[:80]}\n"
        f"[bold]Candidates:[/bold] {candidates_source}\n"
        f"[bold]LLM Provider:[/bold] {LLM_PROVIDER}\n"
        f"[bold]Top N:[/bold] {top_n}",
        title="Pipeline Configuration",
        border_style="cyan",
    ))

    # -----------------------------------------------------------------
    # Step 1: Initialize components
    # -----------------------------------------------------------------
    console.print("\n[bold]━━━ Step 1: Initializing Components ━━━[/bold]\n")

    # Initialize LLM
    llm_client = None
    if use_llm:
        try:
            llm_client = create_llm_client_from_config()
        except Exception as e:
            console.print(f"[yellow]⚠ LLM initialization failed: {e}[/yellow]")
            console.print("[yellow]  Continuing without LLM re-ranking...[/yellow]")

    # Initialize embedding engine
    embedding_engine = EmbeddingEngine()

    # Initialize JD parser
    jd_parser = JDParser(llm_client or create_llm_client_from_config())

    # -----------------------------------------------------------------
    # Step 2: Load and parse data
    # -----------------------------------------------------------------
    console.print("\n[bold]━━━ Step 2: Loading Data ━━━[/bold]\n")

    # Load JD
    jd_path = Path(jd_source)
    if jd_path.exists():
        jd_df = auto_load(jd_path)
        # Assume first row or specific column contains JD text
        # Try common column names
        jd_text_col = None
        for col_name in ["job_description", "description", "jd", "text", "content",
                         "job_desc", "role_description"]:
            for col in jd_df.columns:
                if col.lower().strip() == col_name:
                    jd_text_col = col
                    break
            if jd_text_col:
                break

        if jd_text_col:
            jd_text = str(jd_df[jd_text_col].iloc[0])
        else:
            # Use all columns concatenated
            jd_text = " | ".join(
                f"{col}: {jd_df[col].iloc[0]}"
                for col in jd_df.columns
                if jd_df[col].notna().iloc[0]
            )
        console.print(f"[green]✓ JD loaded from file ({len(jd_text)} chars)[/green]")
    else:
        # JD provided as text
        jd_text = jd_source
        console.print(f"[green]✓ JD provided as text ({len(jd_text)} chars)[/green]")

    # Parse JD
    parsed_jd = jd_parser.parse(jd_text)

    # Load candidates
    candidates_path = Path(candidates_source)
    candidates_df = auto_load(candidates_path)
    profile_dataframe(candidates_df, "Candidates")

    # Parse candidates
    candidate_parser = CandidateParser()
    candidates = candidate_parser.parse_dataframe(candidates_df)

    # -----------------------------------------------------------------
    # Step 3: Run hybrid ranking
    # -----------------------------------------------------------------
    console.print("\n[bold]━━━ Step 3: Running Hybrid Ranking ━━━[/bold]\n")

    ranker = HybridRanker(
        embedding_engine=embedding_engine,
        llm_client=llm_client,
    )

    results = ranker.rank(
        jd=parsed_jd,
        candidates=candidates,
        top_n=top_n,
        use_llm_rerank=use_llm and llm_client is not None,
    )

    # -----------------------------------------------------------------
    # Step 4: Export results
    # -----------------------------------------------------------------
    console.print("\n[bold]━━━ Step 4: Exporting Results ━━━[/bold]\n")

    output_df = format_ranked_output(results, include_rationale=INCLUDE_RATIONALE)
    output_files = export_ranked_candidates(
        output_df,
        output_dir=DATA_OUTPUT_DIR,
        filename="ranked_candidates",
        formats=output_formats,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    console.print("\n" + "═" * 50)
    console.print("[bold green]✅ Pipeline Complete![/bold green]")
    console.print(f"   Candidates evaluated: {len(candidates)}")
    console.print(f"   Top {top_n} ranked and exported")
    for f in output_files:
        console.print(f"   📄 {f}")
    console.print("═" * 50 + "\n")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Candidate Ranking System — INDIA RUNS Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--jd", type=str,
        help="Path to job description file (CSV/Excel/JSON) or raw JD text",
    )
    parser.add_argument(
        "--candidates", type=str,
        help="Path to candidates dataset file",
    )
    parser.add_argument(
        "--jd-text", type=str,
        help="Raw job description text (alternative to --jd file)",
    )
    parser.add_argument(
        "--top-n", type=int, default=OUTPUT_TOP_N,
        help=f"Number of top candidates to output (default: {OUTPUT_TOP_N})",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM re-ranking (faster, less accurate)",
    )
    parser.add_argument(
        "--output-format", type=str, nargs="+", default=["csv"],
        choices=["csv", "xlsx", "json"],
        help="Output format(s) (default: csv)",
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Explore available datasets without running the pipeline",
    )

    args = parser.parse_args()

    print_banner()

    if args.explore:
        explore_data()
        return

    # Determine JD source
    jd_source = args.jd_text or args.jd
    if not jd_source:
        # Try to auto-discover JD file
        jd_files = [f for f in DATA_RAW_DIR.iterdir()
                     if "jd" in f.stem.lower() or "job" in f.stem.lower()]
        if jd_files:
            jd_source = str(jd_files[0])
            console.print(f"[cyan]Auto-detected JD file: {jd_files[0].name}[/cyan]")
        else:
            console.print("[red]✗ No JD source provided. Use --jd or --jd-text[/red]")
            sys.exit(1)

    # Determine candidates source
    candidates_source = args.candidates
    if not candidates_source:
        # Try to auto-discover candidates file
        candidate_files = [f for f in DATA_RAW_DIR.iterdir()
                           if "candidate" in f.stem.lower() or "profile" in f.stem.lower()
                           or "resume" in f.stem.lower()]
        if candidate_files:
            candidates_source = str(candidate_files[0])
            console.print(f"[cyan]Auto-detected candidates file: {candidate_files[0].name}[/cyan]")
        else:
            # Fall back to any remaining data file
            all_files = discover_datasets(DATA_RAW_DIR)
            remaining = [f for f in all_files if str(f) != jd_source]
            if remaining:
                candidates_source = str(remaining[0])
                console.print(f"[cyan]Using candidates file: {remaining[0].name}[/cyan]")
            else:
                console.print("[red]✗ No candidates dataset found. Use --candidates[/red]")
                sys.exit(1)

    run_pipeline(
        jd_source=jd_source,
        candidates_source=candidates_source,
        top_n=args.top_n,
        use_llm=not args.no_llm,
        output_formats=args.output_format,
    )


if __name__ == "__main__":
    main()
