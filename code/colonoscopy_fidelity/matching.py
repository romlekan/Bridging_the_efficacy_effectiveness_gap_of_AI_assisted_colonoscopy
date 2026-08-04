import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from colonoscopy_fidelity.numerics import binary_standardized_difference, standardized_mean_difference


@dataclass(frozen=True)
class Match:
    treated_index: int
    control_index: int
    distance: float


@dataclass(frozen=True)
class BalanceEntry:
    variable: str
    treated_mean: float
    control_mean: float
    treated_sd: float
    control_sd: float
    standardized_difference: float


@dataclass(frozen=True)
class MatchResult:
    matches: Tuple[Match, ...]
    treated_indices: np.ndarray
    control_indices: np.ndarray
    propensity_scores: np.ndarray
    balance_before: Tuple[BalanceEntry, ...]
    balance_after: Tuple[BalanceEntry, ...]


def propensity_scores(features: Sequence[Sequence[float]], treatment: Sequence[int], regularization: float = 1e6, seed: int = 2025) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(treatment, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("features must form a matrix")
    if matrix.shape[0] != labels.size:
        raise ValueError("features and treatment differ in length")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    model = LogisticRegression(C=regularization, solver="lbfgs", max_iter=100000, random_state=seed)
    model.fit(standardized, labels)
    return model.predict_proba(standardized)[:, 1]


def propensity_logit(scores: Sequence[float], epsilon: float = 1e-8) -> np.ndarray:
    array = np.clip(np.asarray(scores, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(array / (1.0 - array))


def common_support(scores: Sequence[float], treatment: Sequence[int]) -> Tuple[float, float]:
    score_array = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(treatment, dtype=np.int64)
    treated = score_array[labels == 1]
    control = score_array[labels == 0]
    if treated.size == 0 or control.size == 0:
        raise ValueError("both treatment groups required")
    return max(float(np.min(treated)), float(np.min(control))), min(float(np.max(treated)), float(np.max(control)))


def support_mask(scores: Sequence[float], treatment: Sequence[int]) -> np.ndarray:
    lower, upper = common_support(scores, treatment)
    array = np.asarray(scores, dtype=np.float64)
    return (array >= lower) & (array <= upper)


def greedy_nearest_matching(scores: Sequence[float], treatment: Sequence[int], caliper: Optional[float] = None, replacement: bool = False) -> Tuple[Match, ...]:
    score_array = propensity_logit(scores)
    labels = np.asarray(treatment, dtype=np.int64)
    treated_indices = np.where(labels == 1)[0]
    control_indices = np.where(labels == 0)[0]
    available = set(int(index) for index in control_indices)
    matches: List[Match] = []
    treated_order = treated_indices[np.argsort(score_array[treated_indices])]
    for treated_index in treated_order:
        candidates = control_indices if replacement else np.asarray(sorted(available), dtype=np.int64)
        if candidates.size == 0:
            break
        distances = np.abs(score_array[candidates] - score_array[treated_index])
        position = int(np.argmin(distances))
        control_index = int(candidates[position])
        distance = float(distances[position])
        if caliper is not None and distance > caliper:
            continue
        matches.append(Match(int(treated_index), control_index, distance))
        if not replacement:
            available.remove(control_index)
    return tuple(matches)


def optimal_pair_matching(scores: Sequence[float], treatment: Sequence[int], caliper: Optional[float] = None) -> Tuple[Match, ...]:
    score_array = propensity_logit(scores)
    labels = np.asarray(treatment, dtype=np.int64)
    treated_indices = np.where(labels == 1)[0]
    control_indices = np.where(labels == 0)[0]
    costs = np.abs(score_array[treated_indices, None] - score_array[control_indices][None, :])
    if caliper is not None:
        costs = np.where(costs <= caliper, costs, 1e9)
    treated_positions, control_positions = optimize.linear_sum_assignment(costs)
    matches: List[Match] = []
    for treated_position, control_position in zip(treated_positions, control_positions):
        distance = float(costs[treated_position, control_position])
        if distance >= 1e9:
            continue
        matches.append(Match(int(treated_indices[treated_position]), int(control_indices[control_position]), distance))
    return tuple(matches)


def exact_stratum_matching(scores: Sequence[float], treatment: Sequence[int], strata: Sequence[object], caliper: Optional[float] = None) -> Tuple[Match, ...]:
    labels = np.asarray(treatment, dtype=np.int64)
    stratum_array = np.asarray(strata, dtype=object)
    matches: List[Match] = []
    for stratum in np.unique(stratum_array):
        indices = np.where(stratum_array == stratum)[0]
        local_labels = labels[indices]
        if np.sum(local_labels == 1) == 0 or np.sum(local_labels == 0) == 0:
            continue
        local_scores = np.asarray(scores, dtype=np.float64)[indices]
        local_matches = optimal_pair_matching(local_scores, local_labels, caliper)
        for item in local_matches:
            matches.append(Match(int(indices[item.treated_index]), int(indices[item.control_index]), item.distance))
    return tuple(matches)


def balance_table(features: Sequence[Sequence[float]], treatment: Sequence[int], names: Sequence[str], treated_indices: Optional[Sequence[int]] = None, control_indices: Optional[Sequence[int]] = None, binary: Sequence[str] = ()) -> Tuple[BalanceEntry, ...]:
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(treatment, dtype=np.int64)
    if treated_indices is None:
        treated = matrix[labels == 1]
    else:
        treated = matrix[np.asarray(treated_indices, dtype=np.int64)]
    if control_indices is None:
        control = matrix[labels == 0]
    else:
        control = matrix[np.asarray(control_indices, dtype=np.int64)]
    entries: List[BalanceEntry] = []
    for index, name in enumerate(names):
        treated_mean = float(np.mean(treated[:, index]))
        control_mean = float(np.mean(control[:, index]))
        treated_sd = float(np.std(treated[:, index], ddof=1))
        control_sd = float(np.std(control[:, index], ddof=1))
        difference = binary_standardized_difference(treated_mean, control_mean) if name in binary else standardized_mean_difference(treated_mean, control_mean, treated_sd, control_sd)
        entries.append(BalanceEntry(name, treated_mean, control_mean, treated_sd, control_sd, difference))
    return tuple(entries)


def match_analysis(features: Sequence[Sequence[float]], treatment: Sequence[int], names: Sequence[str], caliper: float = 0.2, binary: Sequence[str] = (), seed: int = 2025) -> MatchResult:
    scores = propensity_scores(features, treatment, seed=seed)
    logit_sd = float(np.std(propensity_logit(scores), ddof=1))
    matches = optimal_pair_matching(scores, treatment, caliper * logit_sd)
    treated_indices = np.asarray([match.treated_index for match in matches], dtype=np.int64)
    control_indices = np.asarray([match.control_index for match in matches], dtype=np.int64)
    before = balance_table(features, treatment, names, binary=binary)
    after = balance_table(features, treatment, names, treated_indices, control_indices, binary)
    return MatchResult(matches, treated_indices, control_indices, scores, before, after)


def inverse_probability_weights(scores: Sequence[float], treatment: Sequence[int], stabilized: bool = True, trim: Optional[float] = None) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(treatment, dtype=np.int64)
    probability = float(np.mean(labels))
    if stabilized:
        weights = labels * probability / score_array + (1 - labels) * (1.0 - probability) / (1.0 - score_array)
    else:
        weights = labels / score_array + (1 - labels) / (1.0 - score_array)
    if trim is not None:
        lower = float(np.quantile(weights, trim))
        upper = float(np.quantile(weights, 1.0 - trim))
        weights = np.clip(weights, lower, upper)
    return weights


def overlap_weights(scores: Sequence[float], treatment: Sequence[int]) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(treatment, dtype=np.int64)
    return labels * (1.0 - score_array) + (1 - labels) * score_array


def effective_sample_size(weights: Sequence[float]) -> float:
    array = np.asarray(weights, dtype=np.float64)
    return float(np.sum(array) ** 2 / np.sum(array * array))


def weighted_difference(outcome: Sequence[float], treatment: Sequence[int], weights: Sequence[float]) -> float:
    outcome_array = np.asarray(outcome, dtype=np.float64)
    labels = np.asarray(treatment, dtype=np.int64)
    weight_array = np.asarray(weights, dtype=np.float64)
    treated = np.average(outcome_array[labels == 1], weights=weight_array[labels == 1])
    control = np.average(outcome_array[labels == 0], weights=weight_array[labels == 0])
    return float(treated - control)


def matched_difference(outcome: Sequence[float], matches: Sequence[Match]) -> Tuple[float, float]:
    outcome_array = np.asarray(outcome, dtype=np.float64)
    differences = np.asarray([outcome_array[item.treated_index] - outcome_array[item.control_index] for item in matches], dtype=np.float64)
    estimate = float(np.mean(differences))
    error = float(np.std(differences, ddof=1) / math.sqrt(differences.size)) if differences.size > 1 else math.nan
    return estimate, error


def matching_summary(result: MatchResult) -> Mapping[str, object]:
    return {
        "matched_pairs": len(result.matches),
        "mean_distance": float(np.mean([item.distance for item in result.matches])) if result.matches else math.nan,
        "maximum_distance": float(np.max([item.distance for item in result.matches])) if result.matches else math.nan,
        "maximum_smd_before": max(abs(item.standardized_difference) for item in result.balance_before),
        "maximum_smd_after": max(abs(item.standardized_difference) for item in result.balance_after),
    }
