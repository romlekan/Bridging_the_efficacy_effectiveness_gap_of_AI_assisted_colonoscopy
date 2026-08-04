import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from colonoscopy_fidelity.contracts import AnalysisConfig, CohortRecord, ComputeConfig, Interval, OutcomeRecord


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def analysis_config(path: Path) -> AnalysisConfig:
    raw = read_yaml(path).get("analysis", {})
    return AnalysisConfig(
        seed=int(raw.get("seed", 2025)),
        confidence=float(raw.get("confidence", 0.95)),
        alpha=float(raw.get("alpha", 0.05)),
        secondary_alpha=float(raw.get("secondary_alpha", 0.007)),
        rct_benchmark_pp=float(raw.get("rct_benchmark_pp", 8.1)),
        meaningful_gain_pp=float(raw.get("meaningful_gain_pp", 5.0)),
        spline_knots=tuple(float(item) for item in raw.get("spline_knots", [0.25, 0.5, 0.75])),
        fidelity_cutoff=float(raw.get("fidelity_cutoff", 75.0)),
        sensitivity_cutoff=float(raw.get("high_fidelity_sensitivity_cutoff", 80.0)),
        bootstrap_repetitions=int(raw.get("bootstrap_repetitions", 200000)),
    )


def compute_config(path: Path) -> ComputeConfig:
    raw = read_yaml(path)
    compute = raw.get("compute", {})
    network = raw.get("network", {})
    return ComputeConfig(
        world_size=int(compute["world_size"]),
        batch_size=int(compute["batch_size"]),
        gradient_accumulation=int(compute["gradient_accumulation"]),
        epochs=int(compute["epochs"]),
        learning_rate=float(compute["learning_rate"]),
        weight_decay=float(compute["weight_decay"]),
        hidden_width=int(network["hidden_width"]),
        hidden_depth=int(network["hidden_depth"]),
        ensemble_members=int(network["ensemble_members"]),
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    return float(value)


def load_cohorts(path: Path) -> List[CohortRecord]:
    records: List[CohortRecord] = []
    for row in read_rows(path):
        records.append(
            CohortRecord(
                cohort_id=int(row["cohort_id"]),
                group=row["group"],
                sample_size=int(row["n"]),
                adr_percent=float(row["adr_percent"]),
                fidelity_score=optional_float(row.get("fidelity_score")),
                volume_tier=row["volume_tier"],
                annual_volume=optional_float(row.get("annual_volume")),
                baseline_adr=optional_float(row.get("baseline_adr")),
                mean_age=optional_float(row.get("mean_age")),
                female_fraction=optional_float(row.get("female_fraction")),
                screening_fraction=optional_float(row.get("screening_fraction")),
                adequate_prep_fraction=optional_float(row.get("adequate_prep_fraction")),
            )
        )
    return records


def load_outcomes(path: Path) -> List[OutcomeRecord]:
    records: List[OutcomeRecord] = []
    for row in read_rows(path):
        low = optional_float(row.get("ci_low"))
        high = optional_float(row.get("ci_high"))
        interval = Interval(low, high) if low is not None and high is not None else None
        records.append(
            OutcomeRecord(
                name=row["outcome"],
                implementation_value=float(row["implementation_value"]),
                unsupported_value=float(row["unsupported_value"]),
                difference=float(row["difference"]),
                adjusted_odds_ratio=optional_float(row.get("adjusted_odds_ratio")),
                confidence_interval=interval,
            )
        )
    return records


def ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_text(path: Path, text: str) -> None:
    ensure_output(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=json_default))


def json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported JSON value {type(value)!r}")


def write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    ensure_output(path.parent)
    if len(records) == 0:
        atomic_text(path, "")
        return
    keys = list(records[0].keys())
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def select_rows(records: Iterable[Mapping[str, object]], key: str, value: object) -> List[Mapping[str, object]]:
    return [record for record in records if record.get(key) == value]


def require_columns(records: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    if not records:
        raise ValueError("records cannot be empty")
    missing = [column for column in columns if column not in records[0]]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")


def numeric_column(records: Sequence[Mapping[str, object]], name: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(name)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def group_rows(records: Iterable[Mapping[str, object]], key: str) -> Dict[object, List[Mapping[str, object]]]:
    groups: Dict[object, List[Mapping[str, object]]] = {}
    for record in records:
        value = record.get(key)
        groups.setdefault(value, []).append(record)
    return groups
