from __future__ import annotations

import numpy as np


METRIC_COLUMNS = [
    "mean_error_m",
    "mse_m2",
    "median_error_m",
    "rmse_m",
    "cdf_1m",
    "cdf_2m",
    "p90_error_m",
    "p95_error_m",
    "p99_error_m",
    "valid_sample_count",
]


def localization_metrics(errors: np.ndarray) -> dict[str, float | int]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {column: (0 if column == "valid_sample_count" else float("nan")) for column in METRIC_COLUMNS}
    squared = errors**2
    return {
        "mean_error_m": float(np.mean(errors)),
        "mse_m2": float(np.mean(squared)),
        "median_error_m": float(np.median(errors)),
        "rmse_m": float(np.sqrt(np.mean(squared))),
        "cdf_1m": float(np.mean(errors <= 1.0)),
        "cdf_2m": float(np.mean(errors <= 2.0)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "p95_error_m": float(np.percentile(errors, 95)),
        "p99_error_m": float(np.percentile(errors, 99)),
        "valid_sample_count": int(errors.size),
    }
