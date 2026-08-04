from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ResidualSummary:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    skewness: float
    kurtosis: float
    shapiro_statistic: float
    shapiro_p_value: float


@dataclass(frozen=True)
class InfluencePoint:
    index: int
    leverage: float
    studentized_residual: float
    cooks_distance: float
    dffits: float


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    predicted_mean: float
    observed_mean: float
    absolute_error: float


@dataclass(frozen=True)
class CalibrationSummary:
    intercept: float
    slope: float
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    bins: Tuple[CalibrationBin, ...]


def residual_summary(residuals: Sequence[float]) -> ResidualSummary:
    values = np.asarray(residuals, dtype=np.float64)
    shapiro = stats.shapiro(values)
    return ResidualSummary(
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values, ddof=1)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        skewness=float(stats.skew(values, bias=False)),
        kurtosis=float(stats.kurtosis(values, fisher=True, bias=False)),
        shapiro_statistic=float(shapiro.statistic),
        shapiro_p_value=float(shapiro.pvalue),
    )


def hat_matrix(design: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    matrix = np.asarray(design, dtype=np.float64)
    if weights is None:
        weighted = matrix
        square_root = np.eye(matrix.shape[0], dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        square_root = np.diag(np.sqrt(weight_array))
        weighted = square_root @ matrix
    inverse = np.linalg.pinv(weighted.T @ weighted)
    return square_root @ matrix @ inverse @ matrix.T @ square_root


def leverage(design: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    return np.diag(hat_matrix(design, weights))


def studentized_residuals(residuals: Sequence[float], design: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    hat = leverage(design, weights)
    degrees = max(1, values.size - np.asarray(design).shape[1])
    variance = float(np.sum(values * values) / degrees)
    return values / np.sqrt(np.maximum(variance * (1.0 - hat), 1e-15))


def cooks_distance(residuals: Sequence[float], design: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(residuals, dtype=np.float64)
    hat = leverage(matrix, weights)
    degrees = max(1, values.size - matrix.shape[1])
    variance = float(np.sum(values * values) / degrees)
    return values * values * hat / (matrix.shape[1] * variance * np.maximum((1.0 - hat) ** 2, 1e-15))


def influence_points(residuals: Sequence[float], design: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None) -> Tuple[InfluencePoint, ...]:
    matrix = np.asarray(design, dtype=np.float64)
    hat = leverage(matrix, weights)
    studentized = studentized_residuals(residuals, matrix, weights)
    cooks = cooks_distance(residuals, matrix, weights)
    dffits = studentized * np.sqrt(hat / np.maximum(1.0 - hat, 1e-15))
    return tuple(
        InfluencePoint(index, float(hat[index]), float(studentized[index]), float(cooks[index]), float(dffits[index]))
        for index in range(matrix.shape[0])
    )


def condition_indices(design: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(design, dtype=np.float64)
    scaled = matrix / np.sqrt(np.sum(matrix * matrix, axis=0, keepdims=True))
    singular = np.linalg.svd(scaled, compute_uv=False)
    return singular[0] / singular


def variance_inflation_factors(design: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(design, dtype=np.float64)
    values = np.empty(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        outcome = matrix[:, index]
        predictors = np.delete(matrix, index, axis=1)
        predictors = np.column_stack([np.ones(matrix.shape[0]), predictors])
        fitted, _, _, _ = np.linalg.lstsq(predictors, outcome, rcond=None)
        residuals = outcome - predictors @ fitted
        total = np.sum((outcome - np.mean(outcome)) ** 2)
        r_squared = 1.0 - np.sum(residuals * residuals) / total if total > 0.0 else 0.0
        values[index] = 1.0 / max(1e-15, 1.0 - r_squared)
    return values


def calibration_bins(probability: Sequence[float], outcome: Sequence[float], bins: int = 10) -> Tuple[CalibrationBin, ...]:
    predictions = np.asarray(probability, dtype=np.float64)
    outcomes = np.asarray(outcome, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    records: List[CalibrationBin] = []
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (predictions >= lower) & (predictions < upper if index < bins - 1 else predictions <= upper)
        if not np.any(selected):
            continue
        predicted_mean = float(np.mean(predictions[selected]))
        observed_mean = float(np.mean(outcomes[selected]))
        records.append(CalibrationBin(lower, upper, int(np.sum(selected)), predicted_mean, observed_mean, abs(predicted_mean - observed_mean)))
    return tuple(records)


def calibration_summary(probability: Sequence[float], outcome: Sequence[float], bins: int = 10) -> CalibrationSummary:
    predictions = np.clip(np.asarray(probability, dtype=np.float64), 1e-10, 1.0 - 1e-10)
    outcomes = np.asarray(outcome, dtype=np.float64)
    logits = np.log(predictions / (1.0 - predictions))
    design = np.column_stack([np.ones(predictions.size), logits])
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(100):
        linear = design @ coefficients
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-10)
        adjusted = linear + (outcomes - fitted) / weights
        updated = np.linalg.solve(design.T @ (design * weights[:, None]), design.T @ (weights * adjusted))
        if np.max(np.abs(updated - coefficients)) < 1e-12:
            coefficients = updated
            break
        coefficients = updated
    records = calibration_bins(predictions, outcomes, bins)
    total = sum(record.count for record in records)
    expected = sum(record.count * record.absolute_error for record in records) / total
    maximum = max(record.absolute_error for record in records)
    brier = float(np.mean((predictions - outcomes) ** 2))
    return CalibrationSummary(float(coefficients[0]), float(coefficients[1]), expected, maximum, brier, records)


def hosmer_lemeshow(probability: Sequence[float], outcome: Sequence[float], groups: int = 10) -> Mapping[str, float]:
    predictions = np.asarray(probability, dtype=np.float64)
    outcomes = np.asarray(outcome, dtype=np.float64)
    quantiles = np.quantile(predictions, np.linspace(0.0, 1.0, groups + 1))
    statistic = 0.0
    actual_groups = 0
    for index in range(groups):
        selected = (predictions >= quantiles[index]) & (predictions <= quantiles[index + 1] if index == groups - 1 else predictions < quantiles[index + 1])
        count = int(np.sum(selected))
        if count == 0:
            continue
        observed = float(np.sum(outcomes[selected]))
        expected = float(np.sum(predictions[selected]))
        denominator = max(expected * (1.0 - expected / count), 1e-15)
        statistic += (observed - expected) ** 2 / denominator
        actual_groups += 1
    degrees = max(1, actual_groups - 2)
    return {"statistic": statistic, "degrees_freedom": float(degrees), "p_value": float(stats.chi2.sf(statistic, degrees))}


def residual_autocorrelation(residuals: Sequence[float], lag: int = 1) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    if lag <= 0 or lag >= values.size:
        raise ValueError("lag outside valid range")
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def durbin_watson(residuals: Sequence[float]) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    return float(np.sum(np.diff(values) ** 2) / np.sum(values * values))
