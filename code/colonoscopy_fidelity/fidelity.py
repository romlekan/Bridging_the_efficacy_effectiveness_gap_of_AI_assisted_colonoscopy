from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import interpolate, optimize, stats
from sklearn.metrics import auc, roc_curve

from colonoscopy_fidelity.contracts import FidelityComponents, Interval
from colonoscopy_fidelity.numerics import empirical_interval, finite_difference, spearman


@dataclass(frozen=True)
class RubricItem:
    name: str
    maximum: float
    observed: float

    def validate(self) -> None:
        if self.maximum <= 0.0:
            raise ValueError("maximum must be positive")
        if self.observed < 0.0 or self.observed > self.maximum:
            raise ValueError("observed score outside bounds")

    def fraction(self) -> float:
        self.validate()
        return self.observed / self.maximum


@dataclass(frozen=True)
class ComponentRubric:
    name: str
    items: Tuple[RubricItem, ...]
    weight: float = 33.3333333333

    def raw_fraction(self) -> float:
        maximum = sum(item.maximum for item in self.items)
        observed = sum(item.observed for item in self.items)
        if maximum == 0.0:
            raise ValueError("rubric has no possible points")
        for item in self.items:
            item.validate()
        return observed / maximum

    def weighted_score(self) -> float:
        return self.raw_fraction() * self.weight


@dataclass(frozen=True)
class FidelityRubric:
    training: ComponentRubric
    feedback: ComponentRubric
    alert_management: ComponentRubric

    def components(self) -> FidelityComponents:
        return FidelityComponents(
            training=self.training.weighted_score(),
            feedback=self.feedback.weighted_score(),
            alert_management=self.alert_management.weighted_score(),
        )

    def total(self) -> float:
        return self.components().total()


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    sensitivity: float
    specificity: float
    youden_index: float
    auroc: float
    interval: Optional[Interval] = None


@dataclass(frozen=True)
class DoseResponseResult:
    grid: np.ndarray
    prediction: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ed50: float
    minimum_effective: float
    plateau: float
    midrange_slope: float


@dataclass(frozen=True)
class DecompositionResult:
    training_effect: float
    feedback_effect: float
    alert_effect: float
    residual_effect: float
    total_effect: float

    def shares(self) -> Dict[str, float]:
        if self.total_effect == 0.0:
            return {"training": 0.0, "feedback": 0.0, "alert_management": 0.0, "detection": 0.0}
        return {
            "training": 100.0 * self.training_effect / self.total_effect,
            "feedback": 100.0 * self.feedback_effect / self.total_effect,
            "alert_management": 100.0 * self.alert_effect / self.total_effect,
            "detection": 100.0 * self.residual_effect / self.total_effect,
        }


def default_training_rubric(curriculum: float, supervised_cases: float, competency: float) -> ComponentRubric:
    return ComponentRubric(
        name="training",
        items=(
            RubricItem("curriculum", 1.0, curriculum),
            RubricItem("supervised_cases", 1.0, supervised_cases),
            RubricItem("competency", 1.0, competency),
        ),
    )


def default_feedback_rubric(frequency: float, peers: float, dashboard: float) -> ComponentRubric:
    return ComponentRubric(
        name="feedback",
        items=(
            RubricItem("frequency", 1.0, frequency),
            RubricItem("peer_comparison", 1.0, peers),
            RubricItem("dashboard", 1.0, dashboard),
        ),
    )


def default_alert_rubric(calibration: float, timeout: float, adherence: float) -> ComponentRubric:
    return ComponentRubric(
        name="alert_management",
        items=(
            RubricItem("calibration", 1.0, calibration),
            RubricItem("timeout", 1.0, timeout),
            RubricItem("adherence", 1.0, adherence),
        ),
    )


def classify_fidelity(score: float, cutoff: float = 75.0) -> str:
    return "high" if score >= cutoff else "low"


def implementation_ready(score: float, threshold: float = 72.0) -> bool:
    return score >= threshold


def fidelity_stage(score: float) -> str:
    if score < 58.0:
        return "below_effective"
    if score < 72.0:
        return "developing"
    if score < 85.0:
        return "ready"
    return "plateau"


def optimal_threshold(scores: Sequence[float], improvements: Sequence[float], meaningful: float = 5.0) -> ThresholdResult:
    score_array = np.asarray(scores, dtype=np.float64)
    outcome_array = (np.asarray(improvements, dtype=np.float64) >= meaningful).astype(np.int64)
    false_positive, true_positive, thresholds = roc_curve(outcome_array, score_array)
    youden = true_positive - false_positive
    index = int(np.argmax(youden))
    return ThresholdResult(
        threshold=float(thresholds[index]),
        sensitivity=float(true_positive[index]),
        specificity=float(1.0 - false_positive[index]),
        youden_index=float(youden[index]),
        auroc=float(auc(false_positive, true_positive)),
    )


def bootstrap_threshold(scores: Sequence[float], improvements: Sequence[float], meaningful: float, repetitions: int, seed: int) -> ThresholdResult:
    score_array = np.asarray(scores, dtype=np.float64)
    improvement_array = np.asarray(improvements, dtype=np.float64)
    if score_array.size != improvement_array.size:
        raise ValueError("scores and improvements differ in length")
    base = optimal_threshold(score_array, improvement_array, meaningful)
    generator = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(repetitions):
        indices = generator.integers(0, score_array.size, score_array.size)
        labels = improvement_array[indices] >= meaningful
        if len(np.unique(labels)) < 2:
            continue
        estimates.append(optimal_threshold(score_array[indices], improvement_array[indices], meaningful).threshold)
    interval = empirical_interval(estimates) if estimates else None
    return ThresholdResult(base.threshold, base.sensitivity, base.specificity, base.youden_index, base.auroc, interval)


def truncated_power(value: np.ndarray, knot: float, degree: int = 3) -> np.ndarray:
    return np.maximum(value - knot, 0.0) ** degree


def restricted_cubic_basis(values: Sequence[float], knots: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    knot_array = np.asarray(knots, dtype=np.float64)
    if knot_array.size < 3:
        raise ValueError("at least three knots required")
    if np.any(np.diff(knot_array) <= 0.0):
        raise ValueError("knots must increase")
    lower = knot_array[-2]
    upper = knot_array[-1]
    columns = [x]
    for knot in knot_array[:-2]:
        first = truncated_power(x, knot)
        second = truncated_power(x, lower) * (upper - knot) / (upper - lower)
        third = truncated_power(x, upper) * (lower - knot) / (upper - lower)
        columns.append(first - second + third)
    return np.column_stack(columns)


def spline_knots(values: Sequence[float], probabilities: Sequence[float] = (0.25, 0.5, 0.75)) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=np.float64), np.asarray(probabilities, dtype=np.float64))


def fit_restricted_cubic_spline(scores: Sequence[float], improvements: Sequence[float], confidence: float = 0.95, grid_size: int = 1001) -> DoseResponseResult:
    score_array = np.asarray(scores, dtype=np.float64)
    improvement_array = np.asarray(improvements, dtype=np.float64)
    knots = spline_knots(score_array)
    basis = restricted_cubic_basis(score_array, knots)
    design = np.column_stack([np.ones(score_array.size), basis])
    coefficients, _, _, _ = np.linalg.lstsq(design, improvement_array, rcond=None)
    residuals = improvement_array - design @ coefficients
    degrees = max(1, score_array.size - design.shape[1])
    residual_variance = float(np.sum(residuals * residuals) / degrees)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    grid = np.linspace(float(np.min(score_array)), float(np.max(score_array)), grid_size)
    grid_design = np.column_stack([np.ones(grid.size), restricted_cubic_basis(grid, knots)])
    prediction = grid_design @ coefficients
    prediction_variance = np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design)
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, degrees))
    error = np.sqrt(np.maximum(prediction_variance, 0.0))
    lower = prediction - critical * error
    upper = prediction + critical * error
    half_maximum = float(np.min(prediction) + 0.5 * (np.max(prediction) - np.min(prediction)))
    ed50 = float(grid[int(np.argmin(np.abs(prediction - half_maximum)))])
    effective_candidates = grid[lower > 0.0]
    minimum_effective = float(effective_candidates[0]) if effective_candidates.size else float("nan")
    slopes = finite_difference(grid, prediction) * 10.0
    plateau_candidates = grid[slopes < 0.5]
    plateau = float(plateau_candidates[0]) if plateau_candidates.size else float("nan")
    midrange_index = int(np.argmin(np.abs(grid - np.median(score_array))))
    return DoseResponseResult(grid, prediction, lower, upper, ed50, minimum_effective, plateau, float(slopes[midrange_index]))


def monotone_spline(scores: Sequence[float], improvements: Sequence[float], grid_size: int = 1001) -> Tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    improvement_array = np.asarray(improvements, dtype=np.float64)
    order = np.argsort(score_array)
    interpolator = interpolate.PchipInterpolator(score_array[order], improvement_array[order], extrapolate=False)
    grid = np.linspace(float(np.min(score_array)), float(np.max(score_array)), grid_size)
    return grid, interpolator(grid)


def hill_curve(score: np.ndarray, maximum: float, midpoint: float, exponent: float, baseline: float) -> np.ndarray:
    powered = np.power(np.maximum(score, 0.0), exponent)
    midpoint_powered = midpoint**exponent
    return baseline + maximum * powered / (midpoint_powered + powered)


def fit_hill_curve(scores: Sequence[float], improvements: Sequence[float]) -> Tuple[float, float, float, float]:
    score_array = np.asarray(scores, dtype=np.float64)
    improvement_array = np.asarray(improvements, dtype=np.float64)
    initial = [float(np.max(improvement_array)), float(np.median(score_array)), 2.0, float(np.min(improvement_array))]
    bounds = ([0.0, 1.0, 0.1, -20.0], [30.0, 100.0, 10.0, 20.0])
    parameters, _ = optimize.curve_fit(hill_curve, score_array, improvement_array, p0=initial, bounds=bounds, maxfev=100000)
    return tuple(float(value) for value in parameters)


def component_decomposition(training: float, feedback: float, alert: float, total: float) -> DecompositionResult:
    residual = total - training - feedback - alert
    return DecompositionResult(training, feedback, alert, residual, total)


def paper_decomposition() -> DecompositionResult:
    return component_decomposition(0.8, 2.1, 1.4, 5.8)


def fidelity_correlation(scores: Sequence[float], improvements: Sequence[float]) -> Mapping[str, float]:
    coefficient, p_value = spearman(scores, improvements)
    return {"coefficient": coefficient, "p_value": p_value, "n": float(len(scores))}


def gap_recovery(improvement: float, benchmark: float = 8.1) -> float:
    if benchmark == 0.0:
        raise ZeroDivisionError("benchmark cannot be zero")
    return 100.0 * improvement / benchmark


def residual_gap(improvement: float, benchmark: float = 8.1) -> float:
    return benchmark - improvement


def component_priority(result: DecompositionResult) -> List[Tuple[str, float]]:
    shares = result.shares()
    return sorted(shares.items(), key=lambda pair: pair[1], reverse=True)


def adherence_distance(training: float, feedback: float, alert: float) -> Mapping[str, float]:
    return {"training": 100.0 - training, "feedback": 100.0 - feedback, "alert_management": 100.0 - alert}


def temporal_attenuation(initial: float, later: float) -> float:
    if initial == 0.0:
        raise ZeroDivisionError("initial effect cannot be zero")
    return 100.0 * (initial - later) / initial


def maintenance_retention(initial: float, later: float) -> float:
    if initial == 0.0:
        raise ZeroDivisionError("initial effect cannot be zero")
    return 100.0 * later / initial
