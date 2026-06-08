# =============================================================================
# INDIA RUNS — AI Candidate Ranking System
# Data Loader — Flexible dataset ingestion
# =============================================================================

import pandas as pd
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()


def load_csv(filepath: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a CSV file with automatic encoding detection fallback."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        console.print(f"[yellow]UTF-8 failed, trying latin-1 encoding...[/yellow]")
        df = pd.read_csv(filepath, encoding="latin-1")

    console.print(f"[green]✓ Loaded {len(df)} rows from {filepath.name}[/green]")
    return df


def load_excel(filepath: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load an Excel file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
    console.print(f"[green]✓ Loaded {len(df)} rows from {filepath.name}[/green]")
    return df


def load_json(filepath: str | Path) -> list[dict]:
    """Load a JSON file (array of objects or JSONL)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    # Try standard JSON first
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            console.print(f"[green]✓ Loaded {len(data)} records from {filepath.name}[/green]")
            return data
        return [data]
    except json.JSONDecodeError:
        pass

    # Try JSONL (one JSON object per line)
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    console.print(f"[green]✓ Loaded {len(records)} records from {filepath.name} (JSONL)[/green]")
    return records


def auto_load(filepath: str | Path) -> pd.DataFrame:
    """Automatically detect file format and load as DataFrame."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        return load_csv(filepath)
    elif suffix in (".xlsx", ".xls"):
        return load_excel(filepath)
    elif suffix in (".json", ".jsonl"):
        data = load_json(filepath)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def discover_datasets(data_dir: str | Path) -> list[Path]:
    """Discover all dataset files in a directory."""
    data_dir = Path(data_dir)
    supported = {".csv", ".xlsx", ".xls", ".json", ".jsonl"}
    files = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in supported])

    if not files:
        console.print(f"[red]✗ No datasets found in {data_dir}[/red]")
        console.print("[yellow]Please download the dataset from the challenge page "
                      "and place it in data/raw/[/yellow]")
    else:
        console.print(f"\n[bold]📂 Discovered {len(files)} dataset(s):[/bold]")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            console.print(f"   • {f.name} ({size_mb:.1f} MB)")

    return files


def profile_dataframe(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print a rich summary of a DataFrame for exploration."""
    console.print(f"\n[bold cyan]═══ {name} Profile ═══[/bold cyan]")
    console.print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    table = Table(title="Column Summary")
    table.add_column("Column", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Non-Null", justify="right")
    table.add_column("Nulls", justify="right", style="red")
    table.add_column("Unique", justify="right")
    table.add_column("Sample Value", max_width=50)

    for col in df.columns:
        non_null = df[col].notna().sum()
        nulls = df[col].isna().sum()
        unique = df[col].nunique()
        sample = str(df[col].dropna().iloc[0]) if non_null > 0 else "—"
        if len(sample) > 50:
            sample = sample[:47] + "..."

        table.add_row(
            col,
            str(df[col].dtype),
            str(non_null),
            str(nulls) if nulls > 0 else "0",
            str(unique),
            sample,
        )

    console.print(table)
