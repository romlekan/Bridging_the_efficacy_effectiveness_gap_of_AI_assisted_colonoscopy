import hashlib
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from colonoscopy_fidelity.contracts import CohortRecord
from colonoscopy_fidelity.fidelity import gap_recovery, paper_decomposition, residual_gap
from colonoscopy_fidelity.io import atomic_json, load_cohorts, load_outcomes, read_rows, write_csv
from colonoscopy_fidelity.pooling import gee_exchangeable, group_heterogeneity, leave_one_out, logistic_aggregate, pooled_comparison, summarize_cohorts
from colonoscopy_fidelity.reaim import paper_profile, readiness
from colonoscopy_fidelity.sensitivity import quantitative_bias_summary, temporal_decay


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def manifest(paths: Iterable[Path], root: Path) -> List[Mapping[str, object]]:
    records: List[Mapping[str, object]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return records


def environment_record() -> Mapping[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def encode(value: Any) -> Any:
    if is_dataclass(value):
        return {key: encode(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def primary_results(cohorts: Sequence[CohortRecord]) -> Mapping[str, object]:
    comparison = pooled_comparison(cohorts)
    regression = logistic_aggregate(cohorts)
    gee = gee_exchangeable(cohorts)
    implementation_heterogeneity = group_heterogeneity(cohorts, "implementation")
    unsupported_heterogeneity = group_heterogeneity(cohorts, "unsupported")
    return encode(
        {
            "pooled": comparison,
            "aggregate_logistic": regression,
            "gee": gee,
            "implementation_heterogeneity": implementation_heterogeneity,
            "unsupported_heterogeneity": unsupported_heterogeneity,
            "cohort_summary": summarize_cohorts(cohorts),
        }
    )


def paper_targets() -> Mapping[str, object]:
    profile = paper_profile()
    decomposition = paper_decomposition()
    return encode(
        {
            "primary_adr_difference_pp": 5.8,
            "primary_adjusted_odds_ratio": 1.28,
            "primary_interval": [1.11, 1.47],
            "primary_p_value": 0.001,
            "number_needed_to_treat": 17,
            "cohort_tau_squared": 0.013,
            "i_squared": 31.0,
            "fidelity_correlation": 0.71,
            "fidelity_correlation_p": 0.004,
            "threshold": 72.0,
            "threshold_auroc": 0.89,
            "threshold_sensitivity": 0.86,
            "threshold_specificity": 0.83,
            "gap_recovery": gap_recovery(5.8),
            "residual_gap": residual_gap(5.8),
            "reaim": profile.mapping(),
            "readiness": readiness(profile),
            "decomposition": decomposition,
        }
    )


def temporal_results(path: Path) -> Mapping[str, object]:
    rows = read_rows(path)
    complete = [row for row in rows if row["difference"] and row["alert_to_action"]][:4]
    effects = [float(row["difference"]) for row in complete]
    alerts = [float(row["alert_to_action"]) for row in complete]
    return encode(temporal_decay(effects, alerts))


def sensitivity_results(root: Path, cohorts: Sequence[CohortRecord]) -> Mapping[str, object]:
    comparison = pooled_comparison(cohorts)
    return encode(
        {
            "leave_one_out": leave_one_out(cohorts),
            "quantitative_bias": quantitative_bias_summary(comparison.odds_ratio.value),
            "reported_sensitivity": read_rows(root / "data" / "sensitivity.csv"),
            "temporal": temporal_results(root / "data" / "temporal.csv"),
        }
    )


def build_report(root: Path, output: Path) -> Mapping[str, object]:
    cohorts = load_cohorts(root / "data" / "cohorts.csv")
    value = {
        "environment": environment_record(),
        "data_manifest": manifest((root / "data").glob("*.csv"), root),
        "primary": primary_results(cohorts),
        "paper_targets": paper_targets(),
        "sensitivity": sensitivity_results(root, cohorts),
        "outcomes": encode(load_outcomes(root / "data" / "outcomes.csv")),
    }
    atomic_json(output / "analysis.json", value)
    write_csv(output / "cohort_summary.csv", summarize_cohorts(cohorts))
    write_csv(output / "leave_one_out.csv", leave_one_out(cohorts))
    atomic_json(output / "manifest.json", manifest(root.rglob("*"), root))
    return value


def comparison_table(report: Mapping[str, object]) -> List[Mapping[str, object]]:
    targets = report["paper_targets"]
    pooled = report["primary"]["pooled"]
    return [
        {"metric": "ADR difference", "paper": targets["primary_adr_difference_pp"], "computed": pooled["difference"]["value"]},
        {"metric": "odds ratio", "paper": targets["primary_adjusted_odds_ratio"], "computed": pooled["odds_ratio"]["value"]},
        {"metric": "NNT", "paper": targets["number_needed_to_treat"], "computed": pooled["number_needed"]},
    ]
