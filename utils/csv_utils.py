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
    columns_list = list(columns)
    ensure_csv(path, columns_list)
    safe_row = normalise_csv_row(columns_list, row)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns_list)
        writer.writerow(safe_row)

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows if the file exists; otherwise return an empty list."""
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))

def write_csv_rows(path: Path, columns: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to a CSV file, including headers.

    Rows are normalised through `normalise_csv_row` so extra keys are ignored
    and missing keys are written as empty strings.
    """
    columns_list = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns_list)
        writer.writeheader()

        for row in rows:
            writer.writerow(normalise_csv_row(columns_list, row))