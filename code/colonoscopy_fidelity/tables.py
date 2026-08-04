from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Column:
    name: str
    label: str
    kind: str
    decimals: int = 2
    unit: str = ""


@dataclass(frozen=True)
class Cell:
    raw: object
    formatted: str


@dataclass(frozen=True)
class Row:
    key: str
    cells: Tuple[Cell, ...]


@dataclass(frozen=True)
class Table:
    name: str
    columns: Tuple[Column, ...]
    rows: Tuple[Row, ...]

    def records(self) -> List[Mapping[str, object]]:
        values: List[Mapping[str, object]] = []
        for row in self.rows:
            values.append({column.name: cell.raw for column, cell in zip(self.columns, row.cells)})
        return values

    def matrix(self) -> List[List[str]]:
        return [[cell.formatted for cell in row.cells] for row in self.rows]


def format_number(value: object, decimals: int = 2, unit: str = "") -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{numeric:.{decimals}f}{suffix}"


def format_integer(value: object) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def format_p_value(value: object) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if numeric < 0.001:
        return "<0.001"
    if numeric < 0.01:
        return f"{numeric:.3f}"
    return f"{numeric:.2f}"


def format_interval(estimate: object, lower: object, upper: object, decimals: int = 2) -> str:
    if estimate is None or lower is None or upper is None:
        return "—"
    return f"{float(estimate):.{decimals}f} ({float(lower):.{decimals}f}–{float(upper):.{decimals}f})"


def format_cell(value: object, column: Column) -> Cell:
    if column.kind == "integer":
        formatted = format_integer(value)
    elif column.kind == "p_value":
        formatted = format_p_value(value)
    elif column.kind == "text":
        formatted = str(value) if value is not None else "—"
    else:
        formatted = format_number(value, column.decimals, column.unit)
    return Cell(value, formatted)


def build_table(name: str, columns: Sequence[Column], records: Sequence[Mapping[str, object]], key: str) -> Table:
    rows: List[Row] = []
    for index, record in enumerate(records):
        cells = tuple(format_cell(record.get(column.name), column) for column in columns)
        rows.append(Row(str(record.get(key, index)), cells))
    return Table(name, tuple(columns), tuple(rows))


def primary_outcome_columns() -> Tuple[Column, ...]:
    return (
        Column("outcome", "Outcome", "text"),
        Column("implementation_value", "Implementation", "numeric", 1, "%"),
        Column("unsupported_value", "Unsupported", "numeric", 1, "%"),
        Column("difference", "Difference", "numeric", 1, "pp"),
        Column("adjusted_odds_ratio", "aOR", "numeric", 2),
        Column("ci_low", "CI lower", "numeric", 2),
        Column("ci_high", "CI upper", "numeric", 2),
    )


def moderator_columns() -> Tuple[Column, ...]:
    return (
        Column("moderator", "Moderator", "text"),
        Column("subgroup", "Subgroup", "text"),
        Column("improvement", "ADR improvement", "numeric", 1, "pp"),
        Column("ci_low", "CI lower", "numeric", 1),
        Column("ci_high", "CI upper", "numeric", 1),
        Column("interaction_p", "Interaction P", "p_value", 3),
    )


def sensitivity_columns() -> Tuple[Column, ...]:
    return (
        Column("analysis", "Analysis", "text"),
        Column("difference", "ADR difference", "numeric", 1, "pp"),
        Column("odds_ratio", "aOR", "numeric", 2),
        Column("ci_low", "CI lower", "numeric", 2),
        Column("ci_high", "CI upper", "numeric", 2),
        Column("p_value", "P", "p_value", 3),
    )


def temporal_columns() -> Tuple[Column, ...]:
    return (
        Column("period", "Period", "text"),
        Column("implementation_adr", "Implementation ADR", "numeric", 1, "%"),
        Column("unsupported_adr", "Unsupported ADR", "numeric", 1, "%"),
        Column("difference", "Difference", "numeric", 1, "pp"),
        Column("alert_to_action", "Alert to action", "numeric", 1, "%"),
        Column("alerts_ignored", "Alerts ignored", "numeric", 1, "%"),
    )


def table_to_delimited(table: Table, delimiter: str = "\t") -> str:
    header = delimiter.join(column.label for column in table.columns)
    rows = [delimiter.join(cell.formatted for cell in row.cells) for row in table.rows]
    return "\n".join([header] + rows)


def table_to_html(table: Table) -> str:
    header = "".join(f"<th>{column.label}</th>" for column in table.columns)
    body_rows = []
    for row in table.rows:
        cells = "".join(f"<td>{cell.formatted}</td>" for cell in row.cells)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def align_tables(tables: Sequence[Table]) -> Mapping[str, int]:
    widths: Dict[str, int] = {}
    for table in tables:
        for column_index, column in enumerate(table.columns):
            width = len(column.label)
            for row in table.rows:
                width = max(width, len(row.cells[column_index].formatted))
            widths[column.name] = max(widths.get(column.name, 0), width)
    return widths


def fixed_width(table: Table, widths: Optional[Mapping[str, int]] = None) -> str:
    selected = widths if widths is not None else align_tables([table])
    header = "  ".join(column.label.ljust(selected[column.name]) for column in table.columns)
    rows = []
    for row in table.rows:
        rows.append("  ".join(cell.formatted.ljust(selected[column.name]) for column, cell in zip(table.columns, row.cells)))
    return "\n".join([header] + rows)


def numeric_summary(values: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "first_quartile": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "third_quartile": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }
