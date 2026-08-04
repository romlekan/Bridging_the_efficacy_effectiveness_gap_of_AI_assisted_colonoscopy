from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ScaleParameters:
    center: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class ImputationParameters:
    values: np.ndarray
    missing_fraction: np.ndarray


@dataclass(frozen=True)
class TransformBundle:
    imputation: ImputationParameters
    scaling: ScaleParameters
    names: Tuple[str, ...]


def as_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must form a matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix cannot be empty")
    return matrix


def missing_mask(values: Sequence[Sequence[float]]) -> np.ndarray:
    return ~np.isfinite(as_matrix(values))


def fit_median_imputation(values: Sequence[Sequence[float]]) -> ImputationParameters:
    matrix = as_matrix(values)
    medians = np.nanmedian(matrix, axis=0)
    if np.any(~np.isfinite(medians)):
        raise ValueError("column contains no finite values")
    fraction = np.mean(~np.isfinite(matrix), axis=0)
    return ImputationParameters(medians, fraction)


def apply_imputation(values: Sequence[Sequence[float]], parameters: ImputationParameters) -> np.ndarray:
    matrix = as_matrix(values).copy()
    if matrix.shape[1] != parameters.values.size:
        raise ValueError("imputation parameters differ from features")
    rows, columns = np.where(~np.isfinite(matrix))
    matrix[rows, columns] = parameters.values[columns]
    return matrix


def fit_standardization(values: Sequence[Sequence[float]], epsilon: float = 1e-12) -> ScaleParameters:
    matrix = as_matrix(values)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1)
    scale = np.where(scale < epsilon, 1.0, scale)
    return ScaleParameters(center, scale)


def apply_standardization(values: Sequence[Sequence[float]], parameters: ScaleParameters) -> np.ndarray:
    matrix = as_matrix(values)
    if matrix.shape[1] != parameters.center.size:
        raise ValueError("scaling parameters differ from features")
    return (matrix - parameters.center) / parameters.scale


def invert_standardization(values: Sequence[Sequence[float]], parameters: ScaleParameters) -> np.ndarray:
    matrix = as_matrix(values)
    return matrix * parameters.scale + parameters.center


def fit_transform(values: Sequence[Sequence[float]], names: Sequence[str]) -> Tuple[np.ndarray, TransformBundle]:
    matrix = as_matrix(values)
    if matrix.shape[1] != len(names):
        raise ValueError("names differ from features")
    imputation = fit_median_imputation(matrix)
    imputed = apply_imputation(matrix, imputation)
    scaling = fit_standardization(imputed)
    transformed = apply_standardization(imputed, scaling)
    return transformed, TransformBundle(imputation, scaling, tuple(names))


def apply_transform(values: Sequence[Sequence[float]], bundle: TransformBundle) -> np.ndarray:
    return apply_standardization(apply_imputation(values, bundle.imputation), bundle.scaling)


def polynomial_features(values: Sequence[Sequence[float]], degree: int = 2, include_bias: bool = False) -> np.ndarray:
    matrix = as_matrix(values)
    if degree < 1:
        raise ValueError("degree must be positive")
    columns: List[np.ndarray] = []
    if include_bias:
        columns.append(np.ones(matrix.shape[0], dtype=np.float64))
    columns.extend(matrix[:, index] for index in range(matrix.shape[1]))
    if degree >= 2:
        for first in range(matrix.shape[1]):
            for second in range(first, matrix.shape[1]):
                columns.append(matrix[:, first] * matrix[:, second])
    for current_degree in range(3, degree + 1):
        for index in range(matrix.shape[1]):
            columns.append(matrix[:, index] ** current_degree)
    return np.column_stack(columns)


def spline_truncated_power(values: Sequence[float], knots: Sequence[float], degree: int = 3) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    columns = [array**power for power in range(1, degree + 1)]
    columns.extend(np.maximum(array - knot, 0.0) ** degree for knot in knots)
    return np.column_stack(columns)


def winsorize(values: Sequence[float], lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    low = float(np.quantile(array, lower))
    high = float(np.quantile(array, upper))
    return np.clip(array, low, high)


def rank_normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array)
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(1, array.size + 1)
    probabilities = (ranks - 0.5) / array.size
    from scipy.stats import norm
    return norm.ppf(probabilities)


def one_hot(values: Sequence[int], categories: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, Tuple[int, ...]]:
    array = np.asarray(values, dtype=np.int64)
    selected = tuple(int(value) for value in (categories if categories is not None else np.unique(array)))
    matrix = np.column_stack([(array == category).astype(np.float64) for category in selected])
    return matrix, selected


def interaction_terms(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> np.ndarray:
    first_matrix = as_matrix(first)
    second_matrix = as_matrix(second)
    if first_matrix.shape[0] != second_matrix.shape[0]:
        raise ValueError("matrices differ in rows")
    products = [first_matrix[:, first_index] * second_matrix[:, second_index] for first_index in range(first_matrix.shape[1]) for second_index in range(second_matrix.shape[1])]
    return np.column_stack(products)


def concatenate_features(*matrices: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [as_matrix(matrix) for matrix in matrices]
    row_counts = {array.shape[0] for array in arrays}
    if len(row_counts) != 1:
        raise ValueError("feature matrices differ in rows")
    return np.column_stack(arrays)


def feature_summary(values: Sequence[Sequence[float]], names: Sequence[str]) -> List[Mapping[str, float]]:
    matrix = as_matrix(values)
    if matrix.shape[1] != len(names):
        raise ValueError("names differ from columns")
    records: List[Mapping[str, float]] = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        finite = column[np.isfinite(column)]
        records.append(
            {
                "name": name,
                "count": float(finite.size),
                "missing": float(column.size - finite.size),
                "mean": float(np.mean(finite)),
                "standard_deviation": float(np.std(finite, ddof=1)),
                "minimum": float(np.min(finite)),
                "median": float(np.median(finite)),
                "maximum": float(np.max(finite)),
            }
        )
    return records


def correlation_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = as_matrix(values)
    return np.corrcoef(matrix, rowvar=False)


def correlation_filter(values: Sequence[Sequence[float]], threshold: float = 0.95) -> np.ndarray:
    matrix = as_matrix(values)
    correlation = np.abs(correlation_matrix(matrix))
    retained: List[int] = []
    for index in range(matrix.shape[1]):
        if all(correlation[index, existing] < threshold for existing in retained):
            retained.append(index)
    return np.asarray(retained, dtype=np.int64)


def variance_filter(values: Sequence[Sequence[float]], threshold: float = 0.0) -> np.ndarray:
    matrix = as_matrix(values)
    variances = np.var(matrix, axis=0)
    return np.where(variances > threshold)[0]


def select_columns(values: Sequence[Sequence[float]], indices: Sequence[int]) -> np.ndarray:
    matrix = as_matrix(values)
    selected = np.asarray(indices, dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= matrix.shape[1]):
        raise ValueError("column index outside matrix")
    return matrix[:, selected]


def quantile_transform(values: Sequence[float], reference: Optional[Sequence[float]] = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    selected_reference = np.asarray(reference, dtype=np.float64) if reference is not None else np.sort(array)
    ranks = np.searchsorted(np.sort(array), array, side="right")
    probabilities = np.clip((ranks - 0.5) / array.size, 0.0, 1.0)
    positions = probabilities * (selected_reference.size - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    fraction = positions - lower
    return selected_reference[lower] * (1.0 - fraction) + selected_reference[upper] * fraction


def log_transform(values: Sequence[float], offset: Optional[float] = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    selected_offset = offset if offset is not None else max(0.0, 1.0 - float(np.min(array)))
    shifted = array + selected_offset
    if np.any(shifted <= 0.0):
        raise ValueError("log transform received nonpositive values")
    return np.log(shifted)


def bounded_logit(values: Sequence[float], epsilon: float = 1e-6) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(array / (1.0 - array))
