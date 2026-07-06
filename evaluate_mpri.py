from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data import CFRDataset, NormalizeTransform
from src.metrics import METRIC_COLUMNS, localization_metrics
from src.model import MPRINet


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "mpri_eval", "output_root": "runs/eval"},
    "data": {"dataset": "data/processed/front3d_5431_smoke/csi_padded_log_abs_64_snr3.h5", "split": "test"},
    "eval": {"batch_size": 64, "device": "cuda:0", "save_predictions": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MPRI model.")
    parser.add_argument("--config", type=Path, default=Path("config/eval/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("配置文件顶层必须是字典")
    cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    cfg["experiment"]["name"] = str(cfg["experiment"].get("name", "mpri_eval"))
    cfg["experiment"]["output_root"] = str(cfg["experiment"].get("output_root", "runs/eval"))
    cfg["data"]["dataset"] = str(cfg["data"]["dataset"])
    cfg["data"]["split"] = str(cfg["data"].get("split", "test"))
    cfg["eval"]["batch_size"] = int(cfg["eval"].get("batch_size", 64))
    cfg["eval"]["device"] = str(cfg["eval"].get("device", "cuda:0"))
    cfg["eval"]["save_predictions"] = bool(cfg["eval"].get("save_predictions", True))
    return cfg


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.dataset is not None:
        cfg["data"]["dataset"] = str(args.dataset)
    if args.split is not None:
        cfg["data"]["split"] = args.split
    if args.batch_size is not None:
        cfg["eval"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["eval"]["device"] = args.device


def make_run_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["experiment"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    name = cfg["experiment"]["name"]
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = root / f"{name}-{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{name}-{stamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MPRINet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = checkpoint.get("statistics")
    if stats is None:
        raise ValueError("Checkpoint does not contain normalisation statistics.")
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    dropout = float(model_cfg.get("dropout", 0.3))
    output_dim = int(model_cfg.get("output_dim", stats.get("label_dim", 2)))
    in_channels = int(stats["mean"].shape[0])
    model = MPRINet(dropout=dropout, in_channels=in_channels, output_dim=output_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    preds = []
    targets = []
    losses = []
    criterion = nn.MSELoss()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        losses.append(float(criterion(outputs, labels).cpu()))
        preds.append(outputs.cpu())
        targets.append(labels.cpu())
    prediction = torch.cat(preds, dim=0)
    truth = torch.cat(targets, dim=0)
    diff = prediction - truth
    err = torch.linalg.norm(diff, dim=-1)
    err_np = err.numpy()
    metrics = {
        "loss_mse": float(np.mean(losses)) if losses else 0.0,
        **localization_metrics(err_np),
        "axis_mae": torch.mean(torch.abs(diff), dim=0).numpy().tolist(),
    }
    return {
        "metrics": metrics,
        "predictions": prediction.numpy(),
        "ground_truth": truth.numpy(),
        "errors": err.numpy(),
    }


def split_indices(checkpoint: dict[str, Any], split: str, n: int) -> list[int]:
    if split == "all":
        return list(range(n))
    split_info = checkpoint.get("split")
    if not isinstance(split_info, dict):
        raise ValueError("Checkpoint does not contain split information.")
    key = f"{split}_indices"
    if key not in split_info:
        raise ValueError(f"Split '{split}' not found in checkpoint.")
    return [int(v) for v in split_info[key]]


def write_metrics(path: Path, split: str, metrics: dict[str, Any]) -> None:
    row = {"split": split, **{k: v for k, v in metrics.items() if k != "axis_mae"}}
    fieldnames = ["split", "loss_mse", *METRIC_COLUMNS, *[key for key in row if key not in {"split", "loss_mse", *METRIC_COLUMNS}]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_predictions(path: Path, dataset: CFRDataset, indices: list[int], result: dict[str, Any]) -> None:
    gt = result["ground_truth"]
    pred = result["predictions"]
    err = result["errors"]
    global_mobile = dataset.metadata_array("global_mobile_id")
    local_mobile = dataset.metadata_array("local_mobile_id")
    if global_mobile is None:
        global_mobile = dataset.metadata_array("global_tx_id")
    if local_mobile is None:
        local_mobile = dataset.metadata_array("local_tx_id")
    shard = dataset.metadata_array("shard_index")
    source_file = dataset.metadata_array("source_file")
    anchor_ids = dataset.metadata_array("global_anchor_ids")
    valid_anchor_mask = dataset.metadata_array("valid_anchor_mask")
    source_link_ids = dataset.metadata_array("source_link_ids")
    attrs = dataset.attrs()
    fieldnames = [
        "sample_kind",
        "sample_key",
        "sample_index",
        "global_link_id",
        "source_link_ids",
        "global_mobile_id",
        "shard_index",
        "local_mobile_id",
        "bs_id",
        "bs_set_ids",
        "valid_bs_ids",
        "valid_bs_count",
        "localized_side",
        "anchor_side",
        "source_file",
    ]
    dims = gt.shape[1]
    fieldnames.extend([f"true_{axis}" for axis in ["x_m", "y_m", "z_m"][:dims]])
    fieldnames.extend([f"pred_{axis}" for axis in ["x_m", "y_m", "z_m"][:dims]])
    fieldnames.append("error_m")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for out_i, sample_index in enumerate(indices):
            valid_bs_ids = _valid_bs_ids(anchor_ids, valid_anchor_mask, sample_index)
            bs_set_ids = _anchor_ids(anchor_ids)
            sample_kind = "ue_bs_pair" if len(bs_set_ids) == 1 else "ue_bs_set"
            source_ids = _source_link_ids(source_link_ids, sample_index)
            row = {
                "sample_kind": sample_kind,
                "sample_key": _sample_key(global_mobile, sample_index, bs_set_ids, valid_bs_ids),
                "sample_index": sample_index,
                "global_link_id": int(source_ids[0]) if source_ids else -1,
                "source_link_ids": json.dumps(source_ids),
                "global_mobile_id": int(global_mobile[sample_index]) if global_mobile is not None else sample_index,
                "shard_index": int(shard[sample_index]) if shard is not None else "",
                "local_mobile_id": int(local_mobile[sample_index]) if local_mobile is not None else "",
                "bs_id": int(valid_bs_ids[0]) if sample_kind == "ue_bs_pair" and valid_bs_ids else -1,
                "bs_set_ids": json.dumps(bs_set_ids),
                "valid_bs_ids": json.dumps(valid_bs_ids),
                "valid_bs_count": len(valid_bs_ids),
                "localized_side": str(attrs.get("localized_side", "")),
                "anchor_side": str(attrs.get("anchor_side", "")),
                "source_file": _decode(source_file[sample_index]) if source_file is not None else "",
                "error_m": float(err[out_i]),
            }
            for dim, name in enumerate(["x_m", "y_m", "z_m"][:dims]):
                row[f"true_{name}"] = float(gt[out_i, dim])
                row[f"pred_{name}"] = float(pred[out_i, dim])
            writer.writerow(row)


def _anchor_ids(anchor_ids: np.ndarray | None) -> list[int]:
    if anchor_ids is None:
        return []
    return [int(v) for v in np.asarray(anchor_ids).reshape(-1).tolist()]


def _valid_bs_ids(anchor_ids: np.ndarray | None, valid_anchor_mask: np.ndarray | None, sample_index: int) -> list[int]:
    all_anchor_ids = _anchor_ids(anchor_ids)
    if not all_anchor_ids:
        return []
    if valid_anchor_mask is None:
        return all_anchor_ids
    mask = np.asarray(valid_anchor_mask[sample_index]).astype(bool).reshape(-1)
    return [bs for bs, is_valid in zip(all_anchor_ids, mask.tolist(), strict=False) if is_valid]


def _source_link_ids(source_link_ids: np.ndarray | None, sample_index: int) -> list[int]:
    if source_link_ids is None:
        return []
    values = np.asarray(source_link_ids[sample_index]).reshape(-1)
    return [int(v) for v in values.tolist() if int(v) >= 0]


def _sample_key(global_mobile: np.ndarray | None, sample_index: int, bs_set_ids: list[int], valid_bs_ids: list[int]) -> str:
    ue_id = int(global_mobile[sample_index]) if global_mobile is not None else int(sample_index)
    if len(bs_set_ids) == 1:
        bs = valid_bs_ids[0] if valid_bs_ids else bs_set_ids[0]
        return f"ue={ue_id}:bs={bs}"
    return f"ue={ue_id}:bs_set={','.join(str(v) for v in bs_set_ids)}"


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    run_dir = make_run_dir(cfg)
    effective = deepcopy(cfg)
    effective["runtime"] = {
        "checkpoint": str(args.checkpoint),
        "run_dir": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False, allow_unicode=True), encoding="utf-8")

    requested_device = cfg["eval"]["device"]
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    stats = checkpoint["statistics"]
    dataset = CFRDataset(cfg["data"]["dataset"], preload=True)
    dataset.transform = NormalizeTransform(torch.as_tensor(stats["mean"]), torch.as_tensor(stats["std"]))
    indices = split_indices(checkpoint, cfg["data"]["split"], len(dataset))
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg["eval"]["batch_size"], shuffle=False)
    result = evaluate_model(model, loader, device)
    metrics = result["metrics"]
    metrics["axis_mae_json"] = json.dumps(metrics.pop("axis_mae"))
    write_metrics(run_dir / "metrics.csv", cfg["data"]["split"], metrics)
    if cfg["eval"]["save_predictions"]:
        write_predictions(run_dir / "predictions.csv", dataset, indices, result)
    print(json.dumps({"split": cfg["data"]["split"], **metrics}, ensure_ascii=False, indent=2))
    print(run_dir)


if __name__ == "__main__":
    main()
