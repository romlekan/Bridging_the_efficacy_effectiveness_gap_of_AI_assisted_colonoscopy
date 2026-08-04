import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from colonoscopy_fidelity.contracts import CohortRecord, Estimate, Interval
from colonoscopy_fidelity.numerics import cohens_h, e_value_odds_ratio, log_odds_ratio_interval, newcombe_difference_interval, number_needed_to_treat


@dataclass(frozen=True)
class PooledGroup:
    name: str
    events: int
    total: int

    @property
    def rate(self) -> float:
        return self.events / self.total

    @property
    def percent(self) -> float:
        return 100.0 * self.rate


@dataclass(frozen=True)
class PooledComparison:
    implementation: PooledGroup
    unsupported: PooledGroup
    difference: Estimate
    odds_ratio: Estimate
    number_needed: float
    standardized_effect: float
    e_value: float


@dataclass(frozen=True)
class Heterogeneity:
    tau_squared: float
    i_squared: float
    q: float
    degrees_freedom: int
    p_value: float


@dataclass(frozen=True)
class RegressionResult:
    coefficients: Mapping[str, float]
    standard_errors: Mapping[str, float]
    p_values: Mapping[str, float]
    confidence_intervals: Mapping[str, Interval]
    fitted: np.ndarray
    residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float


def aggregate_group(cohorts: Iterable[CohortRecord], name: str) -> PooledGroup:
    selected = [cohort for cohort in cohorts if cohort.group == name]
    if not selected:
        raise ValueError(f"group {name} has no cohorts")
    return PooledGroup(name, sum(cohort.events() for cohort in selected), sum(cohort.sample_size for cohort in selected))


def pooled_comparison(cohorts: Sequence[CohortRecord], confidence: float = 0.95) -> PooledComparison:
    implementation = aggregate_group(cohorts, "implementation")
    unsupported = aggregate_group(cohorts, "unsupported")
    difference_value = implementation.rate - unsupported.rate
    difference_interval = newcombe_difference_interval(
        implementation.events,
        implementation.total,
        unsupported.events,
        unsupported.total,
        confidence,
    )
    ratio, ratio_interval = log_odds_ratio_interval(
        implementation.events,
        implementation.total - implementation.events,
        unsupported.events,
        unsupported.total - unsupported.events,
        confidence,
    )
    table = np.asarray(
        [
            [implementation.events, implementation.total - implementation.events],
            [unsupported.events, unsupported.total - unsupported.events],
        ],
        dtype=np.float64,
    )
    _, p_value = stats.fisher_exact(table)
    difference = Estimate("ADR difference", 100.0 * difference_value, Interval(100.0 * difference_interval.lower, 100.0 * difference_interval.upper, confidence), float(p_value), "percentage points")
    odds = Estimate("ADR odds ratio", ratio, ratio_interval, float(p_value), "ratio")
    return PooledComparison(
        implementation,
        unsupported,
        difference,
        odds,
        number_needed_to_treat(implementation.rate, unsupported.rate),
        cohens_h(implementation.rate, unsupported.rate),
        e_value_odds_ratio(ratio),
    )


def cohort_log_odds(cohort: CohortRecord, correction: float = 0.5) -> Tuple[float, float]:
    events = float(cohort.events())
    non_events = float(cohort.non_events())
    if events == 0.0 or non_events == 0.0:
        events += correction
        non_events += correction
    value = math.log(events / non_events)
    variance = 1.0 / events + 1.0 / non_events
    return value, variance


def heterogeneity(values: Sequence[float], variances: Sequence[float]) -> Heterogeneity:
    value_array = np.asarray(values, dtype=np.float64)
    variance_array = np.asarray(variances, dtype=np.float64)
    weights = 1.0 / variance_array
    center = np.sum(weights * value_array) / np.sum(weights)
    q = float(np.sum(weights * (value_array - center) ** 2))
    degrees = len(values) - 1
    denominator = np.sum(weights) - np.sum(weights * weights) / np.sum(weights)
    tau_squared = max(0.0, (q - degrees) / denominator)
    i_squared = max(0.0, (q - degrees) / q) * 100.0 if q > 0.0 else 0.0
    p_value = float(stats.chi2.sf(q, degrees))
    return Heterogeneity(tau_squared, i_squared, q, degrees, p_value)


def group_heterogeneity(cohorts: Sequence[CohortRecord], group: str) -> Heterogeneity:
    selected = [cohort for cohort in cohorts if cohort.group == group]
    estimates = [cohort_log_odds(cohort) for cohort in selected]
    return heterogeneity([item[0] for item in estimates], [item[1] for item in estimates])


def expand_aggregate_cohorts(cohorts: Sequence[CohortRecord]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for cohort in cohorts:
        events = cohort.events()
        outcomes = np.concatenate([np.ones(events, dtype=np.int8), np.zeros(cohort.sample_size - events, dtype=np.int8)])
        frame = pd.DataFrame(
            {
                "outcome": outcomes,
                "group": np.full(cohort.sample_size, int(cohort.is_supported()), dtype=np.int8),
                "cohort": np.full(cohort.sample_size, cohort.cohort_id, dtype=np.int16),
                "fidelity": np.full(cohort.sample_size, cohort.fidelity_score if cohort.fidelity_score is not None else 0.0),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def design_from_cohorts(cohorts: Sequence[CohortRecord], include_fidelity: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    outcome = np.asarray([cohort.events() / cohort.sample_size for cohort in cohorts], dtype=np.float64)
    group = np.asarray([int(cohort.is_supported()) for cohort in cohorts], dtype=np.float64)
    columns = [np.ones(len(cohorts), dtype=np.float64), group]
    names = ["intercept", "implementation"]
    if include_fidelity:
        fidelity = np.asarray([cohort.fidelity_score or 0.0 for cohort in cohorts], dtype=np.float64)
        columns.append(fidelity / 100.0)
        names.append("fidelity")
    design = np.column_stack(columns)
    return outcome, design, names


def logistic_aggregate(cohorts: Sequence[CohortRecord], include_fidelity: bool = False, confidence: float = 0.95) -> RegressionResult:
    outcome, design, names = design_from_cohorts(cohorts, include_fidelity)
    weights = np.asarray([cohort.sample_size for cohort in cohorts], dtype=np.float64)
    model = sm.GLM(outcome, design, family=sm.families.Binomial(), freq_weights=weights)
    fitted = model.fit()
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    intervals = {name: Interval(float(value - critical * error), float(value + critical * error), confidence) for name, value, error in zip(names, fitted.params, fitted.bse)}
    return RegressionResult(
        coefficients={name: float(value) for name, value in zip(names, fitted.params)},
        standard_errors={name: float(value) for name, value in zip(names, fitted.bse)},
        p_values={name: float(value) for name, value in zip(names, fitted.pvalues)},
        confidence_intervals=intervals,
        fitted=np.asarray(fitted.fittedvalues, dtype=np.float64),
        residuals=np.asarray(fitted.resid_response, dtype=np.float64),
        log_likelihood=float(fitted.llf),
        aic=float(fitted.aic),
        bic=float(fitted.bic_llf),
    )


def gee_exchangeable(cohorts: Sequence[CohortRecord], confidence: float = 0.95) -> RegressionResult:
    frame = pd.DataFrame(
        {
            "outcome": [cohort.events() / cohort.sample_size for cohort in cohorts],
            "group": [int(cohort.is_supported()) for cohort in cohorts],
            "cohort": [cohort.cohort_id for cohort in cohorts],
            "weight": [cohort.sample_size for cohort in cohorts],
        }
    )
    design = sm.add_constant(frame[["group"]], has_constant="add")
    model = sm.GLM(frame["outcome"], design, family=sm.families.Binomial(), freq_weights=frame["weight"])
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": frame["cohort"]})
    names = ["const", "group"]
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    intervals = {name: Interval(float(value - critical * error), float(value + critical * error), confidence) for name, value, error in zip(names, fitted.params, fitted.bse)}
    return RegressionResult(
        coefficients={name: float(value) for name, value in zip(names, fitted.params)},
        standard_errors={name: float(value) for name, value in zip(names, fitted.bse)},
        p_values={name: float(value) for name, value in zip(names, fitted.pvalues)},
        confidence_intervals=intervals,
        fitted=np.asarray(fitted.fittedvalues, dtype=np.float64),
        residuals=np.asarray(frame["outcome"] - fitted.fittedvalues, dtype=np.float64),
        log_likelihood=float("nan"),
        aic=float("nan"),
        bic=float("nan"),
    )


def exponentiate_estimate(result: RegressionResult, name: str) -> Estimate:
    value = math.exp(result.coefficients[name])
    interval = result.confidence_intervals[name]
    return Estimate(name, value, Interval(math.exp(interval.lower), math.exp(interval.upper), interval.level), result.p_values[name], "odds ratio")


def marginal_standardization(result: RegressionResult, group_coefficient: str = "implementation") -> float:
    intercept_name = "intercept" if "intercept" in result.coefficients else "const"
    intercept = result.coefficients[intercept_name]
    coefficient = result.coefficients[group_coefficient]
    unsupported = 1.0 / (1.0 + math.exp(-intercept))
    implementation = 1.0 / (1.0 + math.exp(-(intercept + coefficient)))
    return 100.0 * (implementation - unsupported)


def leave_one_out(cohorts: Sequence[CohortRecord]) -> List[Mapping[str, float]]:
    results: List[Mapping[str, float]] = []
    for omitted in cohorts:
        retained = [cohort for cohort in cohorts if cohort.cohort_id != omitted.cohort_id]
        comparison = pooled_comparison(retained)
        results.append(
            {
                "omitted_cohort": float(omitted.cohort_id),
                "difference_pp": comparison.difference.value,
                "odds_ratio": comparison.odds_ratio.value,
                "nnt": comparison.number_needed,
            }
        )
    return results


def stratified_comparison(cohorts: Sequence[CohortRecord], attribute: str, value: object) -> PooledComparison:
    selected = [cohort for cohort in cohorts if getattr(cohort, attribute) == value]
    return pooled_comparison(selected)


def high_fidelity_subset(cohorts: Sequence[CohortRecord], cutoff: float = 80.0) -> List[CohortRecord]:
    implementation = [cohort for cohort in cohorts if cohort.is_supported() and cohort.fidelity_score is not None and cohort.fidelity_score >= cutoff]
    unsupported = [cohort for cohort in cohorts if not cohort.is_supported()]
    return implementation + unsupported[: len(implementation)]


def summarize_cohorts(cohorts: Sequence[CohortRecord]) -> List[Mapping[str, object]]:
    result: List[Mapping[str, object]] = []
    for group in ("implementation", "unsupported"):
        selected = [cohort for cohort in cohorts if cohort.group == group]
        pooled = aggregate_group(selected, group)
        result.append(
            {
                "group": group,
                "cohorts": len(selected),
                "procedures": pooled.total,
                "events": pooled.events,
                "adr_percent": pooled.percent,
                "mean_cohort_adr": float(np.mean([cohort.adr_percent for cohort in selected])),
                "sd_cohort_adr": float(np.std([cohort.adr_percent for cohort in selected], ddof=1)),
            }
        )
    return result
