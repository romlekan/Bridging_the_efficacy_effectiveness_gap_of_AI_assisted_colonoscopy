import math
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from colonoscopy_fidelity.contracts import Estimate, Interval
from colonoscopy_fidelity.numerics import empirical_interval


@dataclass(frozen=True)
class NegativeControlResult:
    name: str
    treated_rate: float
    control_rate: float
    odds_ratio: float
    interval: Interval
    p_value: float


@dataclass(frozen=True)
class DecayResult:
    initial: float
    final: float
    absolute_change: float
    relative_attenuation: float
    retained_fraction: float
    slope_per_quarter: float
    correlation_with_alert_action: Optional[float]


@dataclass(frozen=True)
class RosenbaumBounds:
    gamma: float
    lower_p: float
    upper_p: float


@dataclass(frozen=True)
class BiasFactor:
    confounder_outcome_risk_ratio: float
    confounder_exposure_risk_ratio: float
    factor: float
    adjusted_risk_ratio: float


def two_by_two_odds_ratio(treated_events: int, treated_total: int, control_events: int, control_total: int, confidence: float = 0.95) -> NegativeControlResult:
    a = float(treated_events)
    b = float(treated_total - treated_events)
    c = float(control_events)
    d = float(control_total - control_events)
    if min(a, b, c, d) == 0.0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    log_ratio = math.log(a * d / (b * c))
    error = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    interval = Interval(math.exp(log_ratio - critical * error), math.exp(log_ratio + critical * error), confidence)
    statistic = log_ratio / error
    p_value = float(2.0 * stats.norm.sf(abs(statistic)))
    return NegativeControlResult("outcome", treated_events / treated_total, control_events / control_total, math.exp(log_ratio), interval, p_value)


def negative_control(name: str, treated_rate: float, treated_total: int, control_rate: float, control_total: int, confidence: float = 0.95) -> NegativeControlResult:
    result = two_by_two_odds_ratio(round(treated_rate * treated_total), treated_total, round(control_rate * control_total), control_total, confidence)
    return NegativeControlResult(name, result.treated_rate, result.control_rate, result.odds_ratio, result.interval, result.p_value)


def temporal_decay(effects: Sequence[float], alert_action: Optional[Sequence[float]] = None) -> DecayResult:
    values = np.asarray(effects, dtype=np.float64)
    if values.size < 2:
        raise ValueError("at least two periods required")
    initial = float(values[0])
    final = float(values[-1])
    absolute_change = initial - final
    relative = 100.0 * absolute_change / initial
    retained = 100.0 * final / initial
    time = np.arange(values.size, dtype=np.float64)
    slope = float(np.polyfit(time, values, 1)[0])
    correlation = None
    if alert_action is not None:
        alerts = np.asarray(alert_action, dtype=np.float64)
        if alerts.size != values.size:
            raise ValueError("alert series length differs")
        correlation = float(stats.pearsonr(values, alerts).statistic)
    return DecayResult(initial, final, absolute_change, relative, retained, slope, correlation)


def nonlinear_decay(time: Sequence[float], initial: float, half_life: float, floor: float) -> np.ndarray:
    values = np.asarray(time, dtype=np.float64)
    return floor + (initial - floor) * np.exp(-math.log(2.0) * values / half_life)


def estimate_half_life(time: Sequence[float], effects: Sequence[float], floor: float = 0.0) -> float:
    time_array = np.asarray(time, dtype=np.float64)
    effect_array = np.asarray(effects, dtype=np.float64)
    shifted = effect_array - floor
    active = shifted > 0.0
    if np.sum(active) < 2:
        return math.nan
    slope, _, _, _, _ = stats.linregress(time_array[active], np.log(shifted[active]))
    if slope >= 0.0:
        return math.inf
    return -math.log(2.0) / slope


def e_value_from_ratio(ratio: float) -> float:
    value = ratio if ratio >= 1.0 else 1.0 / ratio
    return value + math.sqrt(value * (value - 1.0))


def e_value_from_odds_ratio(ratio: float, common_outcome: bool = True) -> float:
    adjusted = math.sqrt(ratio) if common_outcome and ratio >= 1.0 else ratio
    if common_outcome and ratio < 1.0:
        adjusted = math.sqrt(1.0 / ratio)
    return e_value_from_ratio(adjusted)


def bias_factor(confounder_outcome_risk_ratio: float, confounder_exposure_risk_ratio: float, observed_risk_ratio: float) -> BiasFactor:
    numerator = confounder_outcome_risk_ratio * confounder_exposure_risk_ratio
    denominator = confounder_outcome_risk_ratio + confounder_exposure_risk_ratio - 1.0
    factor = numerator / denominator
    return BiasFactor(confounder_outcome_risk_ratio, confounder_exposure_risk_ratio, factor, observed_risk_ratio / factor)


def bias_grid(observed_risk_ratio: float, outcome_strengths: Sequence[float], exposure_strengths: Sequence[float]) -> List[BiasFactor]:
    return [bias_factor(outcome, exposure, observed_risk_ratio) for outcome in outcome_strengths for exposure in exposure_strengths]


def tipping_point(observed_risk_ratio: float, lower: float = 1.0, upper: float = 10.0, steps: int = 10000) -> float:
    strengths = np.linspace(lower, upper, steps)
    adjusted = np.asarray([bias_factor(value, value, observed_risk_ratio).adjusted_risk_ratio for value in strengths])
    indices = np.where(adjusted <= 1.0)[0]
    return float(strengths[indices[0]]) if indices.size else math.nan


def sign_test_bounds(differences: Sequence[float], gamma: float) -> RosenbaumBounds:
    values = np.asarray(differences, dtype=np.float64)
    nonzero = values[values != 0.0]
    positive = int(np.sum(nonzero > 0.0))
    total = nonzero.size
    lower_probability = 1.0 / (1.0 + gamma)
    upper_probability = gamma / (1.0 + gamma)
    lower_p = float(stats.binom.sf(positive - 1, total, upper_probability))
    upper_p = float(stats.binom.sf(positive - 1, total, lower_probability))
    return RosenbaumBounds(gamma, lower_p, upper_p)


def rosenbaum_sequence(differences: Sequence[float], gammas: Sequence[float]) -> Tuple[RosenbaumBounds, ...]:
    return tuple(sign_test_bounds(differences, gamma) for gamma in gammas)


def leave_one_out(values: Sequence[float], statistic: Callable[[np.ndarray], float] = np.mean) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.asarray([statistic(np.delete(array, index)) for index in range(array.size)], dtype=np.float64)


def jackknife_interval(values: Sequence[float], statistic: Callable[[np.ndarray], float] = np.mean, confidence: float = 0.95) -> Interval:
    array = np.asarray(values, dtype=np.float64)
    estimates = leave_one_out(array, statistic)
    center = float(statistic(array))
    pseudo = array.size * center - (array.size - 1) * estimates
    error = float(np.std(pseudo, ddof=1) / math.sqrt(array.size))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, array.size - 1))
    return Interval(center - critical * error, center + critical * error, confidence)


def bootstrap_difference(first: Sequence[float], second: Sequence[float], repetitions: int, seed: int, confidence: float = 0.95) -> Estimate:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        first_sample = generator.choice(first_array, first_array.size, replace=True)
        second_sample = generator.choice(second_array, second_array.size, replace=True)
        estimates[index] = np.mean(first_sample) - np.mean(second_sample)
    interval = empirical_interval(estimates, confidence)
    estimate = float(np.mean(first_array) - np.mean(second_array))
    p_value = 2.0 * min(float(np.mean(estimates <= 0.0)), float(np.mean(estimates >= 0.0)))
    return Estimate("bootstrap difference", estimate, interval, min(1.0, p_value), "")


def placebo_distribution(outcome: Sequence[float], treatment: Sequence[int], repetitions: int, seed: int) -> np.ndarray:
    outcomes = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(treatment, dtype=np.int64)
    generator = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        shuffled = generator.permutation(labels)
        values[index] = float(np.mean(outcomes[shuffled == 1]) - np.mean(outcomes[shuffled == 0]))
    return values


def placebo_p_value(observed: float, distribution: Sequence[float]) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    return float((np.sum(np.abs(values) >= abs(observed)) + 1.0) / (values.size + 1.0))


def calendar_adjustment(years: Sequence[float], effects: Sequence[float]) -> Mapping[str, float]:
    result = stats.linregress(np.asarray(years, dtype=np.float64), np.asarray(effects, dtype=np.float64))
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_value": float(result.rvalue),
        "p_value": float(result.pvalue),
        "standard_error": float(result.stderr),
    }


def worst_case_missing(observed_events: int, observed_total: int, missing: int, event_if_missing: bool) -> float:
    events = observed_events + missing if event_if_missing else observed_events
    return events / (observed_total + missing)


def missing_bounds(observed_events: int, observed_total: int, missing: int) -> Interval:
    return Interval(
        worst_case_missing(observed_events, observed_total, missing, False),
        worst_case_missing(observed_events, observed_total, missing, True),
        1.0,
    )


def quantitative_bias_summary(observed_ratio: float, reported_e_value: float = 1.91) -> Mapping[str, float]:
    return {
        "observed_ratio": observed_ratio,
        "calculated_e_value_common_outcome": e_value_from_odds_ratio(observed_ratio, True),
        "calculated_e_value_rare_outcome": e_value_from_odds_ratio(observed_ratio, False),
        "reported_e_value": reported_e_value,
        "symmetric_tipping_strength": tipping_point(observed_ratio),
    }
