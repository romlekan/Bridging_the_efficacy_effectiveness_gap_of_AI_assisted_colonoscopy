from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats

from colonoscopy_fidelity.contracts import Interval


@dataclass(frozen=True)
class BootstrapResult:
    observed: float
    standard_error: float
    percentile_interval: Interval
    basic_interval: Interval
    normal_interval: Interval
    bias: float
    repetitions: int


@dataclass(frozen=True)
class BcaResult:
    observed: float
    interval: Interval
    bias_correction: float
    acceleration: float
    adjusted_lower_probability: float
    adjusted_upper_probability: float


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    exceedances: int
    repetitions: int
    null_mean: float
    null_standard_deviation: float


@dataclass(frozen=True)
class PowerResult:
    effect: float
    sample_size: int
    power: float
    alpha: float
    repetitions: int


def generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_indices(size: int, repetitions: int, seed: int) -> Iterator[np.ndarray]:
    random = generator(seed)
    for _ in range(repetitions):
        yield random.integers(0, size, size=size)


def ordinary_bootstrap(values: Sequence[float], statistic: Callable[[np.ndarray], float], repetitions: int, seed: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index, indices in enumerate(sample_indices(array.size, repetitions, seed)):
        estimates[index] = statistic(array[indices])
    return estimates


def stratified_bootstrap(values: Sequence[float], strata: Sequence[object], statistic: Callable[[np.ndarray], float], repetitions: int, seed: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    stratum_array = np.asarray(strata, dtype=object)
    random = generator(seed)
    groups = [np.where(stratum_array == value)[0] for value in np.unique(stratum_array)]
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = np.concatenate([random.choice(group, size=group.size, replace=True) for group in groups])
        estimates[repetition] = statistic(array[indices])
    return estimates


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[object], statistic: Callable[[np.ndarray], float], repetitions: int, seed: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    cluster_array = np.asarray(clusters, dtype=object)
    unique = np.unique(cluster_array)
    random = generator(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected = random.choice(unique, size=unique.size, replace=True)
        samples = [array[cluster_array == cluster] for cluster in selected]
        estimates[repetition] = statistic(np.concatenate(samples))
    return estimates


def paired_bootstrap(first: Sequence[float], second: Sequence[float], statistic: Callable[[np.ndarray], float], repetitions: int, seed: int) -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.size != second_array.size:
        raise ValueError("paired samples differ in length")
    return ordinary_bootstrap(first_array - second_array, statistic, repetitions, seed)


def bootstrap_summary(values: Sequence[float], estimates: Sequence[float], confidence: float = 0.95, statistic: Callable[[np.ndarray], float] = np.mean) -> BootstrapResult:
    array = np.asarray(values, dtype=np.float64)
    bootstrap = np.asarray(estimates, dtype=np.float64)
    observed = float(statistic(array))
    alpha = 1.0 - confidence
    lower = float(np.quantile(bootstrap, alpha / 2.0))
    upper = float(np.quantile(bootstrap, 1.0 - alpha / 2.0))
    standard_error = float(np.std(bootstrap, ddof=1))
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return BootstrapResult(
        observed=observed,
        standard_error=standard_error,
        percentile_interval=Interval(lower, upper, confidence),
        basic_interval=Interval(2.0 * observed - upper, 2.0 * observed - lower, confidence),
        normal_interval=Interval(observed - critical * standard_error, observed + critical * standard_error, confidence),
        bias=float(np.mean(bootstrap) - observed),
        repetitions=bootstrap.size,
    )


def jackknife(values: Sequence[float], statistic: Callable[[np.ndarray], float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.asarray([statistic(np.delete(array, index)) for index in range(array.size)], dtype=np.float64)


def acceleration(jackknife_estimates: Sequence[float]) -> float:
    values = np.asarray(jackknife_estimates, dtype=np.float64)
    center = float(np.mean(values))
    differences = center - values
    numerator = np.sum(differences**3)
    denominator = 6.0 * np.sum(differences**2) ** 1.5
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def bca_interval(values: Sequence[float], estimates: Sequence[float], statistic: Callable[[np.ndarray], float], confidence: float = 0.95) -> BcaResult:
    array = np.asarray(values, dtype=np.float64)
    bootstrap = np.asarray(estimates, dtype=np.float64)
    observed = float(statistic(array))
    proportion = (np.sum(bootstrap < observed) + 0.5 * np.sum(bootstrap == observed)) / bootstrap.size
    proportion = min(max(proportion, 1e-10), 1.0 - 1e-10)
    bias_correction = float(stats.norm.ppf(proportion))
    acceleration_value = acceleration(jackknife(array, statistic))
    alpha = 1.0 - confidence
    normal_lower = float(stats.norm.ppf(alpha / 2.0))
    normal_upper = float(stats.norm.ppf(1.0 - alpha / 2.0))
    adjusted_lower = float(stats.norm.cdf(bias_correction + (bias_correction + normal_lower) / (1.0 - acceleration_value * (bias_correction + normal_lower))))
    adjusted_upper = float(stats.norm.cdf(bias_correction + (bias_correction + normal_upper) / (1.0 - acceleration_value * (bias_correction + normal_upper))))
    interval = Interval(float(np.quantile(bootstrap, adjusted_lower)), float(np.quantile(bootstrap, adjusted_upper)), confidence)
    return BcaResult(observed, interval, bias_correction, acceleration_value, adjusted_lower, adjusted_upper)


def difference_statistic(values: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(values[labels == 1]) - np.mean(values[labels == 0]))


def permutation_test(values: Sequence[float], labels: Sequence[int], repetitions: int, seed: int) -> PermutationResult:
    array = np.asarray(values, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    observed = difference_statistic(array, label_array)
    random = generator(seed)
    null = np.empty(repetitions, dtype=np.float64)
    exceedances = 0
    for index in range(repetitions):
        shuffled = random.permutation(label_array)
        null[index] = difference_statistic(array, shuffled)
        exceedances += int(abs(null[index]) >= abs(observed))
    p_value = (exceedances + 1.0) / (repetitions + 1.0)
    return PermutationResult(observed, p_value, exceedances, repetitions, float(np.mean(null)), float(np.std(null, ddof=1)))


def randomization_distribution(group_sizes: Tuple[int, int], event_total: int) -> Mapping[int, float]:
    first_size, second_size = group_sizes
    total = first_size + second_size
    lower = max(0, event_total - second_size)
    upper = min(first_size, event_total)
    support = np.arange(lower, upper + 1)
    probabilities = stats.hypergeom.pmf(support, total, event_total, first_size)
    return {int(value): float(probability) for value, probability in zip(support, probabilities)}


def exact_randomization_p_value(first_events: int, first_total: int, second_events: int, second_total: int) -> float:
    table = np.asarray([[first_events, first_total - first_events], [second_events, second_total - second_events]], dtype=np.int64)
    return float(stats.fisher_exact(table, alternative="two-sided").pvalue)


def simulate_binary_power(control_rate: float, treated_rate: float, control_size: int, treated_size: int, alpha: float, repetitions: int, seed: int) -> PowerResult:
    random = generator(seed)
    rejected = 0
    for _ in range(repetitions):
        control_events = int(random.binomial(control_size, control_rate))
        treated_events = int(random.binomial(treated_size, treated_rate))
        p_value = exact_randomization_p_value(treated_events, treated_size, control_events, control_size)
        rejected += int(p_value < alpha)
    return PowerResult(treated_rate - control_rate, control_size + treated_size, rejected / repetitions, alpha, repetitions)


def detectable_difference(control_rate: float, control_size: int, treated_size: int, target_power: float, alpha: float, repetitions: int, seed: int, lower: float = 0.001, upper: float = 0.25, tolerance: float = 0.001) -> float:
    left = lower
    right = upper
    iteration = 0
    while right - left > tolerance and iteration < 32:
        middle = (left + right) / 2.0
        rate = min(control_rate + middle, 0.999)
        result = simulate_binary_power(control_rate, rate, control_size, treated_size, alpha, repetitions, seed + iteration)
        if result.power >= target_power:
            right = middle
        else:
            left = middle
        iteration += 1
    return (left + right) / 2.0


def monte_carlo_standardization(probabilities_treated: Sequence[float], probabilities_control: Sequence[float], repetitions: int, seed: int) -> np.ndarray:
    treated = np.asarray(probabilities_treated, dtype=np.float64)
    control = np.asarray(probabilities_control, dtype=np.float64)
    random = generator(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        treated_outcome = random.binomial(1, treated)
        control_outcome = random.binomial(1, control)
        values[index] = float(np.mean(treated_outcome) - np.mean(control_outcome))
    return values


def bootstrap_correlation(first: Sequence[float], second: Sequence[float], repetitions: int, seed: int, method: str = "spearman") -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.size != second_array.size:
        raise ValueError("arrays differ in length")
    random = generator(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        indices = random.integers(0, first_array.size, first_array.size)
        if method == "spearman":
            values[index] = stats.spearmanr(first_array[indices], second_array[indices]).statistic
        elif method == "pearson":
            values[index] = stats.pearsonr(first_array[indices], second_array[indices]).statistic
        elif method == "kendall":
            values[index] = stats.kendalltau(first_array[indices], second_array[indices]).statistic
        else:
            raise ValueError("unknown correlation method")
    return values


def bootstrap_quantiles(values: Sequence[float], probabilities: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {str(probability): float(np.quantile(array, probability)) for probability in probabilities}


def simulation_summary(values: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "lower": float(np.quantile(array, 0.025)),
        "upper": float(np.quantile(array, 0.975)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "finite_fraction": float(np.mean(np.isfinite(array))),
    }
