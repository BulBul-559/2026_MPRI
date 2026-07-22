from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.model import MPRINet, count_parameters


FORMAL_METRIC_COLUMNS = (
    "count",
    "mean_error_m",
    "mse_error_m2",
    "median_error_m",
    "rmse_error_m",
    "cdf_at_1m",
    "cdf_at_2m",
    "p90_error_m",
    "p95_error_m",
    "p99_error_m",
)

TEST_COHORTS = {
    "held_out_id_test": (42,),
    "held_out_dc4x_ood_test": (43, 44),
    "held_out_dc3x_cross_condition_test": (30, 31, 32, 33, 34),
    "all_heldout_test": (42, 43, 44, 30, 31, 32, 33, 34),
}


def run_dichasus_formal(
    *,
    protocol_path: str | Path,
    support_size: int,
    output_dir: str | Path,
    device: str = "cuda:0",
    epochs: int = 60,
    batch_size: int = 32,
    eval_batch_size: int = 32,
    subcarrier_count: int = 193,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    seed: int = 2026,
    max_query_links: int | None = None,
) -> Path:
    """Train and evaluate MPRI on the locked formal DICHASUS BS0 protocol."""
    if support_size not in {8, 32, 128, 512}:
        raise ValueError("support_size must be one of 8, 32, 128, 512")
    if epochs <= 0 or batch_size <= 1 or eval_batch_size <= 0:
        raise ValueError("epochs/eval_batch_size must be positive and training batch_size must exceed one")
    if not 4 <= subcarrier_count <= 768:
        raise ValueError("subcarrier_count must be in [4, 768]")

    protocol_path = Path(protocol_path).resolve()
    protocol = _read_json(protocol_path)
    manifest_root = protocol_path.parent / "manifests"
    support_manifest_path = manifest_root / f"support_n{support_size:04d}.json"
    support_manifest = _read_json(support_manifest_path)
    _validate_manifest(support_manifest, expected_count=support_size)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "complete.json"
    if completion_path.exists() and _read_json(completion_path).get("status") == "completed":
        return output

    _set_seed(seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    started_at = time.time()

    subcarrier_indices = np.round(np.linspace(0, 767, subcarrier_count)).astype(np.int64)
    train_features, train_labels = _load_support_training_data(
        support_manifest,
        subcarrier_indices=subcarrier_indices,
        seed=seed,
    )
    channel_mean = train_features.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    channel_std = train_features.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    channel_std[channel_std < 1.0e-6] = 1.0
    train_features = (train_features - channel_mean) / channel_std

    validation_name = protocol["stage2"]["single_bs_manifests"]["dc41"]
    validation_manifest_path = manifest_root / f"{validation_name}.json"
    validation_manifest = _read_json(validation_manifest_path)
    _validate_manifest(validation_manifest)
    validation_rows = list(validation_manifest["rows"])
    if max_query_links is not None:
        validation_rows = validation_rows[: int(max_query_links)]
    validation_features, validation_labels = _load_clean_rows(
        validation_manifest,
        validation_rows,
        subcarrier_indices=subcarrier_indices,
    )
    validation_features = (validation_features - channel_mean) / channel_std

    train_dataset = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_features), torch.from_numpy(validation_labels)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MPRINet(in_channels=1, dropout=0.3, output_dim=2).to(torch_device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.3)
    best_val_rmse = math.inf
    best_epoch = 0
    optimizer_steps = 0
    history_rows: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_errors: list[np.ndarray] = []
        train_loss_sum = 0.0
        train_count = 0
        for features, labels in train_loader:
            features = features.to(torch_device)
            labels = labels.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            radial_sq = torch.sum((predictions - labels) ** 2, dim=-1)
            loss = torch.sqrt(radial_sq.mean() + 1.0e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            train_loss_sum += float(loss.detach()) * int(labels.shape[0])
            train_count += int(labels.shape[0])
            train_errors.append(torch.sqrt(radial_sq.detach()).cpu().numpy())
        scheduler.step()
        validation_errors = _predict_loader(model, validation_loader, torch_device)
        validation_metrics = summarize_errors(validation_errors)
        train_metrics = summarize_errors(np.concatenate(train_errors))
        history_rows.append(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss_sum / max(train_count, 1),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        if float(validation_metrics["rmse_error_m"]) < best_val_rmse:
            best_val_rmse = float(validation_metrics["rmse_error_m"])
            best_epoch = epoch
            _save_checkpoint(
                best_checkpoint,
                model=model,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                channel_mean=channel_mean,
                channel_std=channel_std,
                subcarrier_indices=subcarrier_indices,
                support_size=support_size,
            )

    training_wall_clock_s = time.perf_counter() - training_started
    _save_checkpoint(
        checkpoint_dir / "last.pt",
        model=model,
        epoch=epochs,
        optimizer_steps=optimizer_steps,
        channel_mean=channel_mean,
        channel_std=channel_std,
        subcarrier_indices=subcarrier_indices,
        support_size=support_size,
    )
    _write_csv(output / "train_history.csv", history_rows)
    checkpoint = torch.load(best_checkpoint, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    prediction_path = output / "predictions.csv"
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    per_run_rows: list[dict[str, Any]] = []
    errors_by_run: dict[int, np.ndarray] = {}
    eval_wall_clock_s = 0.0
    with prediction_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_prediction_fieldnames())
        writer.writeheader()
        for role, run_id, manifest_path in _evaluation_manifests(protocol, manifest_root):
            manifest = _read_json(manifest_path)
            _validate_manifest(manifest)
            rows = list(manifest["rows"])
            if max_query_links is not None:
                rows = rows[: int(max_query_links)]
            run_started = time.perf_counter()
            errors = _evaluate_manifest(
                model=model,
                manifest=manifest,
                rows=rows,
                writer=writer,
                split_role=role,
                support_size=support_size,
                subcarrier_indices=subcarrier_indices,
                channel_mean=channel_mean,
                channel_std=channel_std,
                batch_size=eval_batch_size,
                device=torch_device,
            )
            run_wall_clock_s = time.perf_counter() - run_started
            eval_wall_clock_s += run_wall_clock_s
            errors_by_run[run_id] = errors
            per_run_rows.append(
                {
                    "method": "MPRI",
                    "support_size_ue": support_size,
                    "support_link_count": support_size,
                    "split_role": role,
                    "run_id": run_id,
                    "scene_key": f"dichasus_dcxx_dc{run_id:02d}",
                    "aggregation": "run",
                    "run_count": 1,
                    "wall_clock_s": run_wall_clock_s,
                    **summarize_errors(errors),
                }
            )
    prediction_tmp.replace(prediction_path)
    per_run_path = output / "per_run_metrics.csv"
    aggregate_path = output / "aggregate_metrics.csv"
    _write_csv(per_run_path, per_run_rows)
    _write_csv(aggregate_path, _aggregate_rows(errors_by_run, support_size=support_size))

    contract = {
        "schema_version": "baseline_dichasus_mpri_formal_v1",
        "method": "MPRI",
        "task": "single_bs_link_localization",
        "bs_id": 0,
        "support_size_ue": support_size,
        "support_link_count": support_size,
        "representation": "log_abs",
        "model_subcarriers": 768,
        "selected_subcarriers": subcarrier_count,
        "subcarrier_policy": "uniform",
        "augmentation_snr_db": [50.0, 20.0, 10.0],
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": {"type": "StepLR", "step_size_epochs": 50, "gamma": 0.3},
        "checkpoint_selection": "minimum dc41 validation RMSE",
        "selected_epoch": best_epoch,
        "selected_validation_rmse_m": best_val_rmse,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "support_manifest": str(support_manifest_path),
        "support_manifest_sha256": _sha256(support_manifest_path),
        "query_policy": "exact locked DXT BS0 validation/test manifests",
        "sample_unit": "link",
        "smoke_query_limit_per_run": max_query_links,
    }
    _write_json(output / "run_contract.json", contract)
    cost = {
        "schema_version": "baseline_dichasus_cost_v1",
        "method": "MPRI",
        "target_label_ue_count": support_size,
        "target_label_link_count": support_size,
        "augmentation_views_per_link": 3,
        "optimizer_steps": optimizer_steps,
        "consumed_training_examples": epochs * support_size * 3,
        "consumed_unique_training_links": support_size,
        "trainable_parameters": count_parameters(model),
        "new_environment_model_count": 1,
        "new_environment_database_count": 0,
        "training_wall_clock_s": training_wall_clock_s,
        "evaluation_wall_clock_s": eval_wall_clock_s,
        "total_wall_clock_s": time.time() - started_at,
        "device": str(torch_device),
    }
    _write_json(output / "cost.json", cost)
    _write_json(
        completion_path,
        {
            "schema_version": "baseline_dichasus_completion_v1",
            "status": "completed",
            "method": "MPRI",
            "support_size_ue": support_size,
            "selected_epoch": best_epoch,
            "prediction_count": int(sum(values.size for values in errors_by_run.values())),
            "predictions_csv": str(prediction_path),
            "per_run_metrics_csv": str(per_run_path),
            "aggregate_metrics_csv": str(aggregate_path),
            "best_checkpoint": str(best_checkpoint),
            "run_contract_json": str(output / "run_contract.json"),
            "cost_json": str(output / "cost.json"),
        },
    )
    return output


def summarize_errors(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("errors must be a non-empty finite one-dimensional non-negative array")
    mse = float(np.mean(values**2))
    return {
        "count": int(values.size),
        "mean_error_m": float(np.mean(values)),
        "mse_error_m2": mse,
        "median_error_m": float(np.quantile(values, 0.5, method="linear")),
        "rmse_error_m": float(np.sqrt(mse)),
        "cdf_at_1m": float(np.mean(values <= 1.0)),
        "cdf_at_2m": float(np.mean(values <= 2.0)),
        "p90_error_m": float(np.quantile(values, 0.9, method="linear")),
        "p95_error_m": float(np.quantile(values, 0.95, method="linear")),
        "p99_error_m": float(np.quantile(values, 0.99, method="linear")),
    }


def _load_support_training_data(
    manifest: dict[str, Any], *, subcarrier_indices: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = list(manifest["rows"])
    h5_rows = np.asarray([int(row["h5_row"]) for row in rows], dtype=np.int64)
    with h5py.File(manifest["source_h5"]["path"], "r") as handle:
        cfr = _load_model_cfr(handle, h5_rows)[..., subcarrier_indices]
        labels = _ordered_h5_rows(handle["target_xy"], h5_rows).astype(np.float32)
    rng = np.random.default_rng(seed)
    feature_parts = []
    label_parts = []
    for snr_db in (50.0, 20.0, 10.0):
        noisy = _add_awgn(cfr, snr_db, rng)
        feature_parts.append(_complex_to_mpri(noisy))
        label_parts.append(labels)
    return np.concatenate(feature_parts), np.concatenate(label_parts)


def _load_clean_rows(
    manifest: dict[str, Any], rows: list[dict[str, Any]], *, subcarrier_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h5_rows = np.asarray([int(row["h5_row"]) for row in rows], dtype=np.int64)
    with h5py.File(manifest["source_h5"]["path"], "r") as handle:
        cfr = _load_model_cfr(handle, h5_rows)[..., subcarrier_indices]
        labels = _ordered_h5_rows(handle["target_xy"], h5_rows).astype(np.float32)
    return _complex_to_mpri(cfr), labels


def _complex_to_mpri(cfr: np.ndarray) -> np.ndarray:
    log_abs = 20.0 * np.log10(np.abs(cfr) + 1.0e-8)
    flat = log_abs.reshape(log_abs.shape[0], 32, log_abs.shape[-1])
    return np.transpose(flat, (0, 2, 1))[:, None].astype(np.float32)


def _add_awgn(cfr: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(np.abs(cfr) ** 2, axis=(1, 2, 3), keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(cfr.shape) + 1j * rng.standard_normal(cfr.shape)
    )
    return (cfr + noise).astype(np.complex64)


@torch.no_grad()
def _predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    for features, labels in loader:
        predictions = model(features.to(device))
        parts.append(torch.linalg.norm(predictions - labels.to(device), dim=-1).cpu().numpy())
    return np.concatenate(parts)


@torch.no_grad()
def _evaluate_manifest(
    *,
    model: nn.Module,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    writer: csv.DictWriter,
    split_role: str,
    support_size: int,
    subcarrier_indices: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    errors_out: list[np.ndarray] = []
    with h5py.File(manifest["source_h5"]["path"], "r") as handle:
        for start in range(0, len(rows), batch_size):
            chunk_rows = rows[start : start + batch_size]
            h5_rows = np.asarray([int(row["h5_row"]) for row in chunk_rows], dtype=np.int64)
            cfr = _load_model_cfr(handle, h5_rows)[..., subcarrier_indices]
            features = (_complex_to_mpri(cfr) - channel_mean) / channel_std
            labels = _ordered_h5_rows(handle["target_xy"], h5_rows).astype(np.float32)
            predictions = model(torch.from_numpy(features).to(device)).cpu().numpy()
            errors = np.linalg.norm(predictions - labels, axis=1)
            errors_out.append(errors.astype(np.float64))
            for offset, row in enumerate(chunk_rows):
                writer.writerow(
                    {
                        "method": "MPRI",
                        "support_size_ue": support_size,
                        "support_link_count": support_size,
                        "split_role": split_role,
                        "run_id": int(row["run_id"]),
                        "scene_key": row["scene_key"],
                        "sample_identity": row["sample_identity"],
                        "measurement_id": int(row["measurement_id"]),
                        "bs_id": int(row["bs_id"]),
                        "h5_row": int(row["h5_row"]),
                        "link_id": int(row["link_id"]),
                        "pred_x_m": float(predictions[offset, 0]),
                        "pred_y_m": float(predictions[offset, 1]),
                        "gt_x_m": float(labels[offset, 0]),
                        "gt_y_m": float(labels[offset, 1]),
                        "error_m": float(errors[offset]),
                    }
                )
    return np.concatenate(errors_out)


def _load_model_cfr(handle: h5py.File, rows: np.ndarray) -> np.ndarray:
    if str(handle.attrs.get("contract_version", "")) != "dichasus_dcxx_csi_v1":
        raise ValueError(f"unsupported DICHASUS contract in {handle.filename}")
    model_indices = np.asarray(handle["frequency/model_window_indices"], dtype=np.int64)
    if model_indices.shape != (768,):
        raise ValueError(f"expected 768 model subcarriers in {handle.filename}")
    return _ordered_h5_rows(handle["cfr"], rows)[..., model_indices]


def _ordered_h5_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    if np.unique(sorted_rows).size != sorted_rows.size:
        raise ValueError("manifest contains duplicate H5 rows")
    sorted_values = np.asarray(dataset[sorted_rows])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return sorted_values[inverse]


def _evaluation_manifests(
    protocol: dict[str, Any], manifest_root: Path
) -> Iterable[tuple[str, int, Path]]:
    for key, name in protocol["stage2"]["single_bs_manifests"].items():
        run_id = int(str(key).removeprefix("dc"))
        yield ("validation" if run_id == 41 else "test"), run_id, manifest_root / f"{name}.json"


def _aggregate_rows(errors_by_run: dict[int, np.ndarray], *, support_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort, run_ids in (("validation", (41,)), *TEST_COHORTS.items()):
        selected = [errors_by_run[run_id] for run_id in run_ids]
        run_metrics = [summarize_errors(values) for values in selected]
        macro = {
            key: float(np.mean([float(record[key]) for record in run_metrics]))
            for key in FORMAL_METRIC_COLUMNS
            if key != "count"
        }
        rows.append(
            {
                "method": "MPRI",
                "support_size_ue": support_size,
                "support_link_count": support_size,
                "cohort": cohort,
                "aggregation": "run_macro",
                "run_count": len(run_ids),
                "count": int(sum(values.size for values in selected)),
                **macro,
            }
        )
        rows.append(
            {
                "method": "MPRI",
                "support_size_ue": support_size,
                "support_link_count": support_size,
                "cohort": cohort,
                "aggregation": "pooled",
                "run_count": len(run_ids),
                **summarize_errors(np.concatenate(selected)),
            }
        )
    return rows


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    optimizer_steps: int,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    subcarrier_indices: np.ndarray,
    support_size: int,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "statistics": {
                "mean": torch.from_numpy(channel_mean.reshape(-1)),
                "std": torch.from_numpy(channel_std.reshape(-1)),
            },
            "subcarrier_indices": torch.from_numpy(subcarrier_indices),
            "support_size": support_size,
        },
        tmp,
    )
    tmp.replace(path)


def _validate_manifest(manifest: dict[str, Any], expected_count: int | None = None) -> None:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("DICHASUS manifest must contain rows")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"manifest has {len(rows)} rows; expected {expected_count}")
    identities = [str(row["sample_identity"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("DICHASUS manifest has duplicate sample identities")
    if any(int(row["bs_id"]) != 0 for row in rows):
        raise ValueError("formal DICHASUS MPRI task requires BS0")


def _prediction_fieldnames() -> list[str]:
    return [
        "method",
        "support_size_ue",
        "support_link_count",
        "split_role",
        "run_id",
        "scene_key",
        "sample_identity",
        "measurement_id",
        "bs_id",
        "h5_row",
        "link_id",
        "pred_x_m",
        "pred_y_m",
        "gt_x_m",
        "gt_y_m",
        "error_m",
    ]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
