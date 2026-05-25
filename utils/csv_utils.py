import csv
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_csv(path: Path, columns: Iterable[str]) -> None:
    """Create a CSV file with headers if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(columns))
            writer.writeheader()


def normalise_csv_row(columns: Iterable[str], row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a row containing only the configured CSV columns."""
    return {column: row.get(column, "") for column in columns}


def append_csv_row(path: Path, columns: Iterable[str], row: Dict[str, Any]) -> None:
    """Append a single row to a CSV file."""
    ensure_csv(path, columns)
    columns_list = list(columns)
    safe_row = normalise_csv_row(columns_list, row)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns_list)
        writer.writerow(safe_row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv_rows(
    path: Path,
    columns: Iterable[str],
    rows: list[dict[str, Any]],
    *,
    mode: str = "overwrite",
) -> None:
    """Write one or more rows to a CSV file.

    mode="overwrite" replaces the file contents.
    mode="append" appends rows while preserving existing headers.
    """
    columns_list = list(columns)

    if mode == "append":
        for row in rows:
            append_csv_row(path, columns_list, row)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns_list)
        writer.writeheader()

        for row in rows:
            writer.writerow(normalise_csv_row(columns_list, row))
