import math
from typing import Callable, Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, special, stats

from colonoscopy_fidelity.contracts import Interval


Array = np.ndarray


def as_float_array(values: Sequence[float]) -> Array:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("expected one-dimensional values")
    if result.size == 0:
        raise ValueError("values cannot be empty")
    return result


def finite(values: Sequence[float]) -> Array:
    array = as_float_array(values)
    return array[np.isfinite(array)]


def mean(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    array = finite(values)
    if weights is None:
        return float(np.mean(array))
    weight_array = as_float_array(weights)
    if len(weight_array) != len(array):
        raise ValueError("weights and values differ in length")
    return float(np.average(array, weights=weight_array))


def variance(values: Sequence[float], ddof: int = 1) -> float:
    return float(np.var(finite(values), ddof=ddof))


def standard_deviation(values: Sequence[float], ddof: int = 1) -> float:
    return math.sqrt(variance(values, ddof))


def standard_error(values: Sequence[float]) -> float:
    array = finite(values)
    return float(np.std(array, ddof=1) / math.sqrt(array.size))


def quantile(values: Sequence[float], probability: float) -> float:
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability outside unit interval")
    return float(np.quantile(finite(values), probability))


def quantiles(values: Sequence[float], probabilities: Sequence[float]) -> Array:
    return np.quantile(finite(values), np.asarray(probabilities, dtype=np.float64))


def median(values: Sequence[float]) -> float:
    return float(np.median(finite(values)))


def interquartile_range(values: Sequence[float]) -> float:
    lower, upper = quantiles(values, [0.25, 0.75])
    return float(upper - lower)


def logistic(value: Array) -> Array:
    return special.expit(value)


def logit(value: Array, epsilon: float = 1e-12) -> Array:
    bounded = np.clip(value, epsilon, 1.0 - epsilon)
    return special.logit(bounded)


def odds(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("probability must be strictly between zero and one")
    return probability / (1.0 - probability)


def odds_ratio(first: float, second: float) -> float:
    return odds(first) / odds(second)


def risk_ratio(first: float, second: float) -> float:
    if second == 0.0:
        raise ZeroDivisionError("reference risk is zero")
    return first / second


def risk_difference(first: float, second: float) -> float:
    return first - second


def number_needed_to_treat(first: float, second: float) -> float:
    difference = abs(risk_difference(first, second))
    if difference == 0.0:
        return math.inf
    return 1.0 / difference


def cohens_h(first: float, second: float) -> float:
    return 2.0 * math.asin(math.sqrt(first)) - 2.0 * math.asin(math.sqrt(second))


def normal_interval(value: float, error: float, confidence: float = 0.95) -> Interval:
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    return Interval(value - critical * error, value + critical * error, confidence)


def wilson_interval(events: int, total: int, confidence: float = 0.95) -> Interval:
    if total <= 0:
        raise ValueError("total must be positive")
    probability = events / total
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    denominator = 1.0 + critical * critical / total
    center = (probability + critical * critical / (2.0 * total)) / denominator
    margin = critical * math.sqrt(probability * (1.0 - probability) / total + critical * critical / (4.0 * total * total)) / denominator
    return Interval(center - margin, center + margin, confidence)


def newcombe_difference_interval(first_events: int, first_total: int, second_events: int, second_total: int, confidence: float = 0.95) -> Interval:
    first = first_events / first_total
    second = second_events / second_total
    first_interval = wilson_interval(first_events, first_total, confidence)
    second_interval = wilson_interval(second_events, second_total, confidence)
    lower = first - second - math.sqrt((first - first_interval.lower) ** 2 + (second_interval.upper - second) ** 2)
    upper = first - second + math.sqrt((first_interval.upper - first) ** 2 + (second - second_interval.lower) ** 2)
    return Interval(lower, upper, confidence)


def log_odds_ratio_interval(a: int, b: int, c: int, d: int, confidence: float = 0.95, correction: float = 0.5) -> Tuple[float, Interval]:
    cells = np.asarray([a, b, c, d], dtype=np.float64)
    if np.any(cells == 0.0):
        cells += correction
    estimate = float(math.log((cells[0] * cells[3]) / (cells[1] * cells[2])))
    error = float(math.sqrt(np.sum(1.0 / cells)))
    interval = normal_interval(estimate, error, confidence)
    return math.exp(estimate), Interval(math.exp(interval.lower), math.exp(interval.upper), confidence)


def spearman(values: Sequence[float], outcomes: Sequence[float]) -> Tuple[float, float]:
    result = stats.spearmanr(as_float_array(values), as_float_array(outcomes))
    return float(result.statistic), float(result.pvalue)


def pearson(values: Sequence[float], outcomes: Sequence[float]) -> Tuple[float, float]:
    result = stats.pearsonr(as_float_array(values), as_float_array(outcomes))
    return float(result.statistic), float(result.pvalue)


def kendall(values: Sequence[float], outcomes: Sequence[float]) -> Tuple[float, float]:
    result = stats.kendalltau(as_float_array(values), as_float_array(outcomes))
    return float(result.statistic), float(result.pvalue)


def standardized_mean_difference(first_mean: float, second_mean: float, first_sd: float, second_sd: float) -> float:
    pooled = math.sqrt((first_sd * first_sd + second_sd * second_sd) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (first_mean - second_mean) / pooled


def binary_standardized_difference(first: float, second: float) -> float:
    pooled = math.sqrt((first * (1.0 - first) + second * (1.0 - second)) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (first - second) / pooled


def inverse_variance_weights(variances: Sequence[float], tau_squared: float = 0.0) -> Array:
    array = as_float_array(variances)
    if np.any(array <= 0.0):
        raise ValueError("variances must be positive")
    return 1.0 / (array + tau_squared)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    value_array = as_float_array(values)
    weight_array = as_float_array(weights)
    if value_array.size != weight_array.size:
        raise ValueError("values and weights differ in length")
    return float(np.sum(value_array * weight_array) / np.sum(weight_array))


def cochran_q(values: Sequence[float], variances: Sequence[float]) -> float:
    weights = inverse_variance_weights(variances)
    center = weighted_mean(values, weights)
    array = as_float_array(values)
    return float(np.sum(weights * (array - center) ** 2))


def dersimonian_laird(values: Sequence[float], variances: Sequence[float]) -> float:
    weights = inverse_variance_weights(variances)
    q = cochran_q(values, variances)
    degrees = len(values) - 1
    correction = np.sum(weights) - np.sum(weights * weights) / np.sum(weights)
    return float(max(0.0, (q - degrees) / correction))


def i_squared(values: Sequence[float], variances: Sequence[float]) -> float:
    q = cochran_q(values, variances)
    degrees = len(values) - 1
    if q <= 0.0:
        return 0.0
    return max(0.0, (q - degrees) / q) * 100.0


def random_effects(values: Sequence[float], variances: Sequence[float], confidence: float = 0.95) -> Tuple[float, Interval, float, float]:
    tau_squared = dersimonian_laird(values, variances)
    weights = inverse_variance_weights(variances, tau_squared)
    estimate = weighted_mean(values, weights)
    error = math.sqrt(1.0 / float(np.sum(weights)))
    interval = normal_interval(estimate, error, confidence)
    return estimate, interval, tau_squared, i_squared(values, variances)


def empirical_interval(values: Sequence[float], confidence: float = 0.95) -> Interval:
    alpha = 1.0 - confidence
    array = finite(values)
    return Interval(float(np.quantile(array, alpha / 2.0)), float(np.quantile(array, 1.0 - alpha / 2.0)), confidence)


def bootstrap(values: Sequence[float], statistic: Callable[[Array], float], repetitions: int, seed: int) -> Array:
    array = finite(values)
    generator = np.random.default_rng(seed)
    result = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.choice(array, size=array.size, replace=True)
        result[index] = statistic(sample)
    return result


def paired_bootstrap(first: Sequence[float], second: Sequence[float], statistic: Callable[[Array, Array], float], repetitions: int, seed: int) -> Array:
    first_array = as_float_array(first)
    second_array = as_float_array(second)
    if first_array.size != second_array.size:
        raise ValueError("paired arrays differ in length")
    generator = np.random.default_rng(seed)
    result = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        indices = generator.integers(0, first_array.size, size=first_array.size)
        result[index] = statistic(first_array[indices], second_array[indices])
    return result


def permutation_p_value(first: Sequence[float], second: Sequence[float], repetitions: int, seed: int) -> float:
    first_array = as_float_array(first)
    second_array = as_float_array(second)
    observed = abs(float(np.mean(first_array) - np.mean(second_array)))
    combined = np.concatenate([first_array, second_array])
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repetitions):
        generator.shuffle(combined)
        difference = abs(float(np.mean(combined[: first_array.size]) - np.mean(combined[first_array.size :])))
        exceedances += int(difference >= observed)
    return (exceedances + 1.0) / (repetitions + 1.0)


def e_value(risk_ratio_value: float) -> float:
    if risk_ratio_value < 1.0:
        risk_ratio_value = 1.0 / risk_ratio_value
    return risk_ratio_value + math.sqrt(risk_ratio_value * (risk_ratio_value - 1.0))


def e_value_odds_ratio(odds_ratio_value: float, rare_outcome: bool = False) -> float:
    if odds_ratio_value < 1.0:
        odds_ratio_value = 1.0 / odds_ratio_value
    approximate_rr = odds_ratio_value if rare_outcome else math.sqrt(odds_ratio_value)
    return e_value(approximate_rr)


def bonferroni(alpha: float, comparisons: int) -> float:
    if comparisons <= 0:
        raise ValueError("comparisons must be positive")
    return alpha / comparisons


def benjamini_hochberg(p_values: Sequence[float]) -> Array:
    values = as_float_array(p_values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    return restored


def simpson_integral(x: Sequence[float], y: Sequence[float]) -> float:
    return float(np.trapz(as_float_array(y), as_float_array(x)))


def finite_difference(x: Sequence[float], y: Sequence[float]) -> Array:
    return np.gradient(as_float_array(y), as_float_array(x))


def root_bracket(function: Callable[[float], float], lower: float, upper: float) -> float:
    return float(optimize.brentq(function, lower, upper))


def minimize_scalar(function: Callable[[float], float], lower: float, upper: float) -> float:
    result = optimize.minimize_scalar(function, bounds=(lower, upper), method="bounded")
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x)


def chunks(values: Sequence[float], size: int) -> Iterator[Sequence[float]]:
    if size <= 0:
        raise ValueError("size must be positive")
    for index in range(0, len(values), size):
        yield values[index : index + size]
