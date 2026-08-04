import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from colonoscopy_fidelity.contracts import CohortRecord, OutcomeRecord


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    location: str


@dataclass(frozen=True)
class ValidationReport:
    findings: Tuple[Finding, ...]

    def errors(self) -> Tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "error")

    def warnings(self) -> Tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "warning")

    def valid(self) -> bool:
        return len(self.errors()) == 0

    def raise_for_errors(self) -> None:
        if self.errors():
            messages = "; ".join(finding.message for finding in self.errors())
            raise ValueError(messages)


def validate_cohort(cohort: CohortRecord) -> List[Finding]:
    findings: List[Finding] = []
    location = f"cohort:{cohort.cohort_id}"
    if cohort.cohort_id <= 0:
        findings.append(Finding("cohort_id", "error", "cohort identifier must be positive", location))
    if cohort.group not in {"implementation", "unsupported"}:
        findings.append(Finding("group", "error", "unknown cohort group", location))
    if cohort.sample_size <= 0:
        findings.append(Finding("sample_size", "error", "sample size must be positive", location))
    if cohort.adr_percent <= 0.0 or cohort.adr_percent >= 100.0:
        findings.append(Finding("adr", "error", "ADR must be between zero and one hundred", location))
    if cohort.is_supported() and cohort.fidelity_score is None:
        findings.append(Finding("fidelity_missing", "error", "supported cohort requires fidelity score", location))
    if not cohort.is_supported() and cohort.fidelity_score is not None:
        findings.append(Finding("fidelity_unexpected", "warning", "unsupported cohort has fidelity score", location))
    if cohort.fidelity_score is not None and (cohort.fidelity_score < 0.0 or cohort.fidelity_score > 100.0):
        findings.append(Finding("fidelity_range", "error", "fidelity score outside range", location))
    if cohort.volume_tier not in {"high", "low"}:
        findings.append(Finding("volume_tier", "error", "volume tier must be high or low", location))
    return findings


def validate_cohorts(cohorts: Sequence[CohortRecord]) -> ValidationReport:
    findings: List[Finding] = []
    for cohort in cohorts:
        findings.extend(validate_cohort(cohort))
    identifiers = [cohort.cohort_id for cohort in cohorts]
    if len(set(identifiers)) != len(identifiers):
        findings.append(Finding("duplicate_id", "error", "cohort identifiers are not unique", "cohorts"))
    implementation_count = sum(cohort.is_supported() for cohort in cohorts)
    unsupported_count = len(cohorts) - implementation_count
    if implementation_count != 8:
        findings.append(Finding("implementation_count", "error", "expected eight supported cohorts", "cohorts"))
    if unsupported_count != 8:
        findings.append(Finding("unsupported_count", "error", "expected eight unsupported cohorts", "cohorts"))
    if len(cohorts) != 16:
        findings.append(Finding("cohort_count", "error", "expected sixteen cohorts", "cohorts"))
    return ValidationReport(tuple(findings))


def validate_outcome(outcome: OutcomeRecord) -> List[Finding]:
    findings: List[Finding] = []
    location = f"outcome:{outcome.name}"
    calculated = outcome.implementation_value - outcome.unsupported_value
    if not math.isclose(calculated, outcome.difference, abs_tol=0.11):
        findings.append(Finding("difference", "error", "reported difference does not match group values", location))
    if outcome.adjusted_odds_ratio is not None and outcome.adjusted_odds_ratio <= 0.0:
        findings.append(Finding("odds_ratio", "error", "odds ratio must be positive", location))
    if outcome.confidence_interval is not None:
        if outcome.confidence_interval.lower > outcome.confidence_interval.upper:
            findings.append(Finding("interval_order", "error", "confidence interval is reversed", location))
        if outcome.adjusted_odds_ratio is not None and not outcome.confidence_interval.contains(outcome.adjusted_odds_ratio):
            findings.append(Finding("interval_contains", "warning", "confidence interval excludes point estimate", location))
    return findings


def validate_outcomes(outcomes: Sequence[OutcomeRecord]) -> ValidationReport:
    findings: List[Finding] = []
    for outcome in outcomes:
        findings.extend(validate_outcome(outcome))
    names = [outcome.name for outcome in outcomes]
    if len(set(names)) != len(names):
        findings.append(Finding("duplicate_outcome", "error", "outcome names are not unique", "outcomes"))
    if "ADR" not in names:
        findings.append(Finding("primary_missing", "error", "primary ADR outcome is missing", "outcomes"))
    return ValidationReport(tuple(findings))


def merge_reports(reports: Sequence[ValidationReport]) -> ValidationReport:
    return ValidationReport(tuple(finding for report in reports for finding in report.findings))


def validate_files(paths: Sequence[Path]) -> ValidationReport:
    findings: List[Finding] = []
    for path in paths:
        if not path.exists():
            findings.append(Finding("file_missing", "error", "required file is missing", path.name))
        elif not path.is_file():
            findings.append(Finding("not_file", "error", "required path is not a file", path.name))
        elif path.stat().st_size == 0:
            findings.append(Finding("file_empty", "error", "required file is empty", path.name))
    return ValidationReport(tuple(findings))


def check_target(name: str, observed: float, expected: float, tolerance: float) -> Optional[Finding]:
    if abs(observed - expected) > tolerance:
        return Finding("target_mismatch", "warning", f"{name} differs from target", name)
    return None


def target_report(values: Mapping[str, float], targets: Mapping[str, float], tolerances: Mapping[str, float]) -> ValidationReport:
    findings: List[Finding] = []
    for name, expected in targets.items():
        if name not in values:
            findings.append(Finding("target_absent", "error", f"{name} is absent", name))
            continue
        finding = check_target(name, values[name], expected, tolerances.get(name, 0.0))
        if finding is not None:
            findings.append(finding)
    return ValidationReport(tuple(findings))
