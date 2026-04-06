"""Config-driven benchmark runner for classic, Fusion V2, and Vision V3 experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dvlog_dataset import create_dvlog_dataloaders
from src.data.edaic_dataset import create_edaic_dataloaders
from src.data.fusion_v2_datasets import create_fusion_v2_dataloaders
from src.data.vision_v3_datasets import create_vision_v3_dataloaders
from src.model.encoders import BimodalSequenceClassifier, SequenceBinaryClassifier
from src.model.fusion_v2 import FusionV2Model
from src.model.vision_v3 import VisionV3Model
from src.paths import PROJECT_ROOT, repo_relative
from src.training.fusion_v2 import (
    FusionV2TrainConfig,
    persist_fusion_v2_seed_artifacts,
    set_seed as set_fusion_seed,
    train_fusion_v2_one_seed,
)
from src.training.trainer import TrainConfig, persist_seed_artifacts, set_seed, train_one_seed
from src.training.vision_v3 import (
    VisionV3TrainConfig,
    persist_vision_v3_seed_artifacts,
    set_seed as set_vision_v3_seed,
    train_vision_v3_one_seed,
)

INPUT_DIMS = {
    ("dvlog", "acoustic"): 25,
    ("dvlog", "visual"): 136,
    ("dvlog", "both"): {"visual": 136, "acoustic": 25},
    ("edaic", "acoustic"): 23,
    ("edaic", "visual"): 49,
    ("edaic", "both"): {"visual": 49, "acoustic": 23},
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_selection_ledger(path: str) -> dict[str, Any]:
    ledger_path = _resolve_path(path)
    if not ledger_path.exists():
        raise RuntimeError(f"Selection ledger not found: {ledger_path}")
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _window_stride(window_size: int, policy: str) -> int:
    if policy == "half_window":
        return max(1, window_size // 2)
    if policy == "fixed_15":
        return 15
    raise ValueError(f"Unsupported stride policy: {policy}")


def _model_family(suite: dict[str, Any]) -> str:
    return str(suite.get("model_family", "classic"))


def _build_loaders(track: dict[str, Any], experiment: dict[str, Any], suite: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_family = _model_family(suite)
    if model_family == "fusion_v2":
        loaders = create_fusion_v2_dataloaders(
            dataset_name=track["dataset"],
            bundle_name=experiment.get("bundle_name", track["bundle_name"]),
            batch_size=experiment["batch_size"],
            window_size=experiment["window_size"],
            stride=experiment["stride"],
            num_workers=experiment.get("num_workers", 0),
            balanced_sampling=experiment["balanced_sampling"],
            min_valid_ratio=experiment.get("min_valid_ratio", 0.5),
        )
        metadata = dict(loaders["metadata"])
        return {"train": loaders["train"], "dev": loaders["dev"], "test": loaders["test"]}, metadata

    if model_family == "vision_v3":
        loaders = create_vision_v3_dataloaders(
            bundle_name=experiment.get("bundle_name", track["bundle_name"]),
            batch_size=experiment["batch_size"],
            window_size=experiment["window_size"],
            stride=experiment["stride"],
            num_workers=experiment.get("num_workers", 0),
            balanced_sampling=experiment["balanced_sampling"],
            min_valid_ratio=experiment.get("min_valid_ratio", 0.5),
        )
        metadata = dict(loaders["metadata"])
        return {"train": loaders["train"], "dev": loaders["dev"], "test": loaders["test"]}, metadata

    if track["dataset"] == "dvlog":
        loaders = create_dvlog_dataloaders(
            batch_size=experiment["batch_size"],
            window_size=experiment["window_size"],
            stride=experiment["stride"],
            num_workers=experiment.get("num_workers", 0),
            balanced_sampling=experiment["balanced_sampling"],
            normalization_source=experiment.get("normalization_source", "train"),
        )
        return {"train": loaders["train"], "dev": loaders["valid"], "test": loaders["test"]}, {}

    loaders = create_edaic_dataloaders(
        modality=track["modality"],
        batch_size=experiment["batch_size"],
        window_size=experiment["window_size"],
        stride=experiment["stride"],
        num_workers=experiment.get("num_workers", 0),
        balanced_sampling=experiment["balanced_sampling"],
        min_valid_ratio=experiment.get("min_valid_ratio", 0.5),
    )
    return loaders, {}


def _build_train_config(experiment: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        batch_size=experiment["batch_size"],
        hidden_dim=experiment["hidden_dim"],
        num_layers=experiment["num_layers"],
        dropout=experiment["dropout"],
        learning_rate=experiment["learning_rate"],
        weight_decay=experiment["weight_decay"],
        epochs=experiment["epochs"],
        patience=experiment["patience"],
        device=experiment["device"],
        aggregation_methods=tuple(experiment["aggregation_methods"]),
        loss_name=experiment["loss_name"],
        focal_gamma=experiment.get("focal_gamma", 2.0),
        focal_alpha=experiment.get("focal_alpha", 0.75),
        use_pos_weight=experiment.get("use_pos_weight", False),
    )


def _build_fusion_v2_train_config(experiment: dict[str, Any]) -> FusionV2TrainConfig:
    return FusionV2TrainConfig(
        batch_size=experiment["batch_size"],
        hidden_dim=experiment["hidden_dim"],
        dropout=experiment["dropout"],
        learning_rate=experiment["learning_rate"],
        weight_decay=experiment["weight_decay"],
        epochs=experiment["epochs"],
        patience=experiment["patience"],
        device=experiment["device"],
        freeze_warm_start_epochs=experiment.get("freeze_warm_start_epochs", 3),
        modality_dropout=experiment.get("modality_dropout", 0.15),
        fused_binary_weight=experiment.get("fused_binary_weight", 1.0),
        acoustic_aux_weight=experiment.get("acoustic_aux_weight", 0.30),
        visual_aux_weight=experiment.get("visual_aux_weight", 0.15),
        teacher_kl_weight=experiment.get("teacher_kl_weight", 0.20),
        phq_regression_weight=experiment.get("phq_regression_weight", 0.25),
        phq_severity_weight=experiment.get("phq_severity_weight", 0.10),
        gate_entropy_weight=experiment.get("gate_entropy_weight", 0.02),
    )


def _build_vision_v3_train_config(experiment: dict[str, Any]) -> VisionV3TrainConfig:
    return VisionV3TrainConfig(
        batch_size=experiment["batch_size"],
        hidden_dim=experiment["hidden_dim"],
        dropout=experiment["dropout"],
        learning_rate=experiment["learning_rate"],
        weight_decay=experiment["weight_decay"],
        epochs=experiment["epochs"],
        patience=experiment["patience"],
        device=experiment["device"],
        modality_dropout=experiment.get("modality_dropout", 0.10),
        mixture_binary_weight=experiment.get("mixture_binary_weight", 1.0),
        visual_aux_weight=experiment.get("visual_aux_weight", 0.35),
        acoustic_aux_weight=experiment.get("acoustic_aux_weight", 0.10),
        teacher_kl_weight=experiment.get("teacher_kl_weight", 0.10),
        gate_entropy_weight=experiment.get("gate_entropy_weight", 0.01),
    )


def _instantiate_model(track: dict[str, Any], experiment: dict[str, Any], suite: dict[str, Any], metadata: dict[str, Any]):
    model_family = _model_family(suite)
    if model_family == "fusion_v2":
        return FusionV2Model(
            modality_dims=metadata["modality_dims"],
            modality_groups=metadata["modality_groups"],
            quality_dim=metadata["quality_dim"],
            hidden_dim=experiment["hidden_dim"],
            dropout=experiment["dropout"],
            latent_num=experiment.get("latent_num", 8),
            cross_attn_heads=experiment.get("cross_attn_heads", 4),
            self_attn_layers=experiment.get("self_attn_layers", 4),
            self_attn_heads=experiment.get("self_attn_heads", 4),
            gate_variant=experiment.get("gate_variant", "quality_mixture"),
            subject_aggregation=experiment.get("subject_aggregation", "transformer"),
            modality_dropout=experiment.get("modality_dropout", 0.15),
        )

    if model_family == "vision_v3":
        return VisionV3Model(
            modality_dims=metadata["modality_dims"],
            modality_groups=metadata["modality_groups"],
            quality_dim=metadata["quality_dim"],
            hidden_dim=experiment["hidden_dim"],
            dropout=experiment["dropout"],
            subject_aggregation=experiment.get("subject_aggregation", "transformer"),
            fusion_variant=experiment.get("fusion_variant", "fixed_prior"),
            fixed_prior_weights=tuple(experiment.get("fixed_prior_weights", [0.10, 0.60, 0.30])),
            modality_dropout=experiment.get("modality_dropout", 0.10),
        )

    if track["modality"] == "both":
        dims = INPUT_DIMS[(track["dataset"], track["modality"])]
        return BimodalSequenceClassifier(
            visual_input_dim=dims["visual"],
            acoustic_input_dim=dims["acoustic"],
            hidden_dim=experiment["hidden_dim"],
            num_layers=experiment["num_layers"],
            dropout=experiment["dropout"],
        )

    return SequenceBinaryClassifier(
        input_dim=INPUT_DIMS[(track["dataset"], track["modality"])],
        hidden_dim=experiment["hidden_dim"],
        num_layers=experiment["num_layers"],
        dropout=experiment["dropout"],
    )


def _numeric_summary(metric_rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    keys = set()
    for row in metric_rows:
        keys.update(row.keys())
    for key in sorted(keys):
        values = [row[key] for row in metric_rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        if np.all(np.isnan(arr)):
            mean_metrics[key] = float("nan")
            std_metrics[key] = float("nan")
        else:
            mean_metrics[key] = float(np.nanmean(arr))
            std_metrics[key] = float(np.nanstd(arr))
    return mean_metrics, std_metrics


def _select_aggregation(seed_results: list[dict[str, Any]], methods: list[str]) -> str:
    scores = {}
    for method in methods:
        values = [payload["dev_results"][method]["metrics"]["macro_f1"] for payload in seed_results]
        scores[method] = float(np.mean(values))
    return max(methods, key=lambda method: scores[method])


def _experiment_summary(
    track: dict[str, Any],
    experiment: dict[str, Any],
    stage_name: str,
    candidate_name: str,
    seed_results: list[dict[str, Any]],
    include_test: bool,
    suite: dict[str, Any],
    selected_method: str | None = None,
) -> dict[str, Any]:
    model_family = _model_family(suite)
    if model_family in {"fusion_v2", "vision_v3"}:
        dev_rows = [payload["dev_results"]["metrics"] for payload in seed_results]
        dev_mean, dev_std = _numeric_summary(dev_rows)
        summary = {
            "suite_track": track["name"],
            "model_family": model_family,
            "dataset": track["dataset"],
            "modality": experiment.get("bundle_name", track.get("bundle_name", track.get("modality"))),
            "stage_name": stage_name,
            "candidate_name": candidate_name,
            "completed_at": _timestamp(),
            "selected_aggregation": experiment.get("subject_aggregation", "transformer"),
            "include_test": include_test,
            "experiment_config": experiment,
            "train_config": asdict(
                _build_fusion_v2_train_config(experiment) if model_family == "fusion_v2" else _build_vision_v3_train_config(experiment)
            ),
            "seeds": [],
            "dev_metrics_mean": dev_mean,
            "dev_metrics_std": dev_std,
        }
        for payload in seed_results:
            item = {
                "seed": payload["seed"],
                "best_epoch": payload["best_epoch"],
                "dev_metrics": payload["dev_results"]["metrics"],
                "dev_gate_stats": payload["dev_results"]["gate_stats"],
            }
            if include_test and payload["test_results"] is not None:
                item["test_metrics"] = payload["test_results"]["metrics"]
                item["test_gate_stats"] = payload["test_results"]["gate_stats"]
            summary["seeds"].append(item)
        if include_test:
            test_rows = [payload["test_results"]["metrics"] for payload in seed_results if payload["test_results"] is not None]
            test_mean, test_std = _numeric_summary(test_rows)
            summary["test_metrics_mean"] = test_mean
            summary["test_metrics_std"] = test_std
        return summary

    assert selected_method is not None
    dev_rows = [payload["dev_results"][selected_method]["metrics"] for payload in seed_results]
    dev_mean, dev_std = _numeric_summary(dev_rows)
    summary = {
        "suite_track": track["name"],
        "model_family": model_family,
        "dataset": track["dataset"],
        "modality": track["modality"],
        "stage_name": stage_name,
        "candidate_name": candidate_name,
        "completed_at": _timestamp(),
        "selected_aggregation": selected_method,
        "include_test": include_test,
        "experiment_config": experiment,
        "train_config": asdict(_build_train_config(experiment)),
        "seeds": [],
        "dev_metrics_mean": dev_mean,
        "dev_metrics_std": dev_std,
    }
    for payload in seed_results:
        item = {
            "seed": payload["seed"],
            "best_epoch": payload["best_epoch"],
            "best_dev_method": payload["best_method"],
            "dev_metrics": payload["dev_results"][selected_method]["metrics"],
        }
        if include_test and payload["test_results"] is not None:
            item["test_metrics"] = payload["test_results"][selected_method]["metrics"]
        summary["seeds"].append(item)
    if include_test:
        test_rows = [payload["test_results"][selected_method]["metrics"] for payload in seed_results if payload["test_results"] is not None]
        test_mean, test_std = _numeric_summary(test_rows)
        summary["test_metrics_mean"] = test_mean
        summary["test_metrics_std"] = test_std
    return summary


def _run_experiment(
    track: dict[str, Any],
    experiment: dict[str, Any],
    experiment_dir: Path,
    stage_name: str,
    candidate_name: str,
    seeds: list[int],
    include_test: bool,
    suite: dict[str, Any],
    forced_aggregation: str | None = None,
) -> dict[str, Any]:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    (experiment_dir / "config.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    model_family = _model_family(suite)

    seed_results = []
    loaders, metadata = _build_loaders(track, experiment, suite)
    for seed in seeds:
        if model_family == "fusion_v2":
            set_fusion_seed(seed)
        elif model_family == "vision_v3":
            set_vision_v3_seed(seed)
        else:
            set_seed(seed)
        model = _instantiate_model(track, experiment, suite, metadata)
        if model_family == "fusion_v2":
            result = train_fusion_v2_one_seed(
                model=model,
                train_loader=loaders["train"],
                dev_loader=loaders["dev"],
                test_loader=loaders["test"] if include_test else None,
                config=_build_fusion_v2_train_config(experiment),
                teacher_specs=experiment.get("teacher_specs"),
            )
        elif model_family == "vision_v3":
            result = train_vision_v3_one_seed(
                model=model,
                train_loader=loaders["train"],
                dev_loader=loaders["dev"],
                test_loader=loaders["test"] if include_test else None,
                config=_build_vision_v3_train_config(experiment),
                teacher_specs=experiment.get("teacher_specs"),
            )
        else:
            result = train_one_seed(
                model=model,
                train_loader=loaders["train"],
                dev_loader=loaders["dev"],
                test_loader=loaders["test"] if include_test else None,
                modality=track["modality"],
                config=_build_train_config(experiment),
            )
        seed_results.append({"seed": seed, **result})

    selected_method = None
    if model_family == "classic":
        selected_method = forced_aggregation or _select_aggregation(seed_results, list(experiment["aggregation_methods"]))
    summary = _experiment_summary(
        track=track,
        experiment=experiment,
        stage_name=stage_name,
        candidate_name=candidate_name,
        seed_results=seed_results,
        include_test=include_test,
        suite=suite,
        selected_method=selected_method,
    )

    for payload in seed_results:
        seed_dir = experiment_dir / f"seed_{payload['seed']}"
        if model_family == "fusion_v2":
            persist_fusion_v2_seed_artifacts(seed_dir, payload["seed"], payload)
        elif model_family == "vision_v3":
            persist_vision_v3_seed_artifacts(seed_dir, payload["seed"], payload)
        else:
            persist_seed_artifacts(seed_dir, payload["seed"], payload, selected_method)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _stage_choice(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(candidates, key=lambda item: item["summary"]["dev_metrics_mean"]["macro_f1"])


def _write_selection_ledger(path: Path, ledger: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def _generate_leaderboard(output_root: Path):
    rows = []
    for summary_path in output_root.rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "stage": summary["stage_name"],
                "track": summary["suite_track"],
                "dataset": summary["dataset"],
                "modality": summary["modality"],
                "candidate": summary["candidate_name"],
                "selected_aggregation": summary["selected_aggregation"],
                "dev_macro_f1_mean": summary["dev_metrics_mean"].get("macro_f1"),
                "dev_macro_f1_std": summary["dev_metrics_std"].get("macro_f1"),
                "test_macro_f1_mean": summary.get("test_metrics_mean", {}).get("macro_f1"),
                "test_macro_f1_std": summary.get("test_metrics_std", {}).get("macro_f1"),
                "summary_path": repo_relative(summary_path),
            }
        )
    if rows:
        leaderboard = pd.DataFrame(rows).sort_values(
            by=["stage", "dataset", "modality", "dev_macro_f1_mean"],
            ascending=[True, True, True, False],
        )
        leaderboard.to_csv(output_root / "leaderboard.csv", index=False)


def _collect_final_rows(output_root: Path, ledger: dict[str, Any]) -> list[dict[str, Any]]:
    final_rows = []
    for track_name in ledger["tracks"]:
        final_summary_path = output_root / "final" / track_name / "locked_run" / "summary.json"
        if not final_summary_path.exists():
            continue
        summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        final_rows.append(
            {
                "track": track_name,
                "dataset": summary["dataset"],
                "modality": summary["modality"],
                "selected_aggregation": summary["selected_aggregation"],
                "window_size": summary["experiment_config"]["window_size"],
                "loss_name": summary["experiment_config"].get("loss_name", summary.get("model_family", "fusion_v2")),
                "balanced_sampling": summary["experiment_config"]["balanced_sampling"],
                "use_pos_weight": summary["experiment_config"].get("use_pos_weight", False),
                "hidden_dim": summary["experiment_config"]["hidden_dim"],
                "num_layers": summary["experiment_config"].get("num_layers", summary["experiment_config"].get("self_attn_layers", 0)),
                "normalization_source": summary["experiment_config"].get("normalization_source", "train"),
                "dev_macro_f1_mean": summary["dev_metrics_mean"].get("macro_f1"),
                "dev_macro_f1_std": summary["dev_metrics_std"].get("macro_f1"),
                "test_macro_f1_mean": summary.get("test_metrics_mean", {}).get("macro_f1"),
                "test_macro_f1_std": summary.get("test_metrics_std", {}).get("macro_f1"),
            }
        )
    return final_rows


def _write_final_report(output_root: Path, ledger: dict[str, Any], suite: dict[str, Any]):
    final_rows = _collect_final_rows(output_root, ledger)
    if not final_rows:
        return

    frame = pd.DataFrame(final_rows)
    report_references = suite.get("report_references", [])
    if report_references:
        frame["source"] = suite["suite_name"]
        reference_frames = [frame]
        for item in report_references:
            path = _resolve_path(item["path"])
            if path.exists():
                ref = pd.read_csv(path)
                ref["source"] = item.get("name", repo_relative(path))
                reference_frames.append(ref)
        frame = pd.concat(reference_frames, ignore_index=True, sort=False)

    frame = frame.sort_values(by=["dataset", "test_macro_f1_mean"], ascending=[True, False])
    frame.to_csv(output_root / "final" / "benchmark_summary.csv", index=False)

    headers = list(frame.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        table_lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")

    suite_name = str(ledger.get("suite_name", "benchmark_suite"))
    suite_kind = "benchmark"
    if "unimodal" in suite_name:
        suite_kind = "unimodal"
    elif "bimodal" in suite_name or "multimodal" in suite_name or "fusion" in suite_name:
        suite_kind = "bimodal"

    lines = [
        f"# {suite_kind.capitalize()} Benchmark Report",
        "",
        f"- Generated: {_timestamp()}",
        f"- Suite: {suite_name}",
        "",
        "## Final Baselines",
        "",
        *table_lines,
        "",
        "## Modality Ranking",
        "",
    ]

    for dataset, group in frame.groupby("dataset", sort=True):
        lines.append(f"### {dataset}")
        for _, row in group.sort_values("test_macro_f1_mean", ascending=False).iterrows():
            source = f" [{row['source']}]" if "source" in row and not pd.isna(row["source"]) else ""
            lines.append(
                f"- {row['modality']}{source}: test macro F1 {row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f}"
            )
        lines.append("")

    lines.extend(["## Conclusions", ""])
    if "vision_v3" in suite_name:
        lines.append("- Vision V3 reports prioritize visual-first bundles and track mixture weights, quality slices, and teacher alignment.")
        lines.append("- Promotion should follow the vision-first rule: prefer vision/fusion when the gap to acoustic stays within the accepted margin.")
    elif "fusion_v2" in suite_name:
        lines.append("- Fusion V2 reports now include gate, quality-slice, and teacher-alignment artifacts alongside core metrics.")
        lines.append("- Promotion should still be based on the dev-stage selection rule and the locked showdown stability.")
    elif suite_kind == "unimodal":
        lines.append("- Bimodal work is justified only if the next model is required to beat the stronger unimodal result on each dataset.")
    else:
        lines.append("- Compare each multimodal result against the strongest unimodal baseline for the same dataset before promotion.")
    lines.append("- This report is the source of truth for the finalized benchmark run.")

    report_path = output_root / "final" / "milestone_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _base_experiment_config(suite: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    config = dict(suite["base_experiment"])
    if "aggregation_methods" in config:
        config["aggregation_methods"] = list(config["aggregation_methods"])
    config["normalization_source"] = track.get("default_normalization", config.get("normalization_source", "train"))
    if "base_overrides" in track:
        config = _merge_dicts(config, track["base_overrides"])
    return config


def _record_stage(ledger_track: dict[str, Any], stage_name: str, track_root: Path, stage_candidates: list[dict[str, Any]], selected: str):
    ledger_track["stages"][stage_name] = {
        "candidates": [
            {
                "candidate_name": item["candidate_name"],
                "summary_path": repo_relative(track_root / stage_name / item["candidate_name"] / "summary.json"),
                "dev_macro_f1_mean": item["summary"]["dev_metrics_mean"]["macro_f1"],
                "selected_aggregation": item["summary"]["selected_aggregation"],
            }
            for item in stage_candidates
        ],
        "selected": selected,
    }


def _run_generic_dev_stage(suite: dict[str, Any], output_root: Path):
    ledger_path = output_root / "selection_ledger.json"
    ledger = {"suite_name": suite["suite_name"], "completed_at": None, "tracks": {}}

    for track in suite["tracks"]:
        track_name = track["name"]
        track_root = output_root / "dev" / track_name
        current_choice = {"experiment": _base_experiment_config(suite, track)}
        ledger["tracks"][track_name] = {
            "dataset": track["dataset"],
            "modality": track.get("bundle_name", track.get("modality")),
            "stages": {},
            "final_config": None,
            "frozen_aggregation": None,
        }

        for stage in track["search_stages"]:
            stage_name = stage["name"]
            stage_candidates = []
            for candidate in stage["candidates"]:
                candidate_name = candidate["name"]
                experiment = _merge_dicts(current_choice["experiment"], candidate.get("overrides", {}))
                if "window_size" in experiment and "stride" not in experiment:
                    experiment["stride"] = _window_stride(experiment["window_size"], suite["stride_policy"])
                summary = _run_experiment(
                    track=track,
                    experiment=experiment,
                    experiment_dir=track_root / stage_name / candidate_name,
                    stage_name=stage_name,
                    candidate_name=candidate_name,
                    seeds=suite["dev_seeds"],
                    include_test=False,
                    suite=suite,
                )
                stage_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
            chosen = _stage_choice(stage_candidates)
            _record_stage(ledger["tracks"][track_name], stage_name, track_root, stage_candidates, chosen["candidate_name"])
            _write_selection_ledger(ledger_path, ledger)
            current_choice = chosen

        ledger["tracks"][track_name]["final_config"] = current_choice["experiment"]
        ledger["tracks"][track_name]["frozen_aggregation"] = current_choice["summary"]["selected_aggregation"]
        _write_selection_ledger(ledger_path, ledger)

    ledger["completed_at"] = _timestamp()
    _write_selection_ledger(ledger_path, ledger)
    _generate_leaderboard(output_root)


def _run_classic_dev_stage(suite: dict[str, Any], output_root: Path):
    ledger_path = output_root / "selection_ledger.json"
    ledger = {"suite_name": suite["suite_name"], "completed_at": None, "tracks": {}}

    for track in suite["tracks"]:
        track_name = track["name"]
        track_root = output_root / "dev" / track_name
        base_config = _base_experiment_config(suite, track)
        ledger["tracks"][track_name] = {
            "dataset": track["dataset"],
            "modality": track["modality"],
            "stages": {},
            "final_config": None,
            "frozen_aggregation": None,
        }

        stage_a_candidates = []
        for window_size in track["window_sizes"]:
            candidate_name = f"window_{window_size:02d}"
            experiment = _merge_dicts(base_config, {"window_size": window_size, "stride": _window_stride(window_size, suite["stride_policy"])})
            summary = _run_experiment(track, experiment, track_root / "stage_a" / candidate_name, "stage_a", candidate_name, suite["dev_seeds"], False, suite)
            stage_a_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
        chosen_a = _stage_choice(stage_a_candidates)
        _record_stage(ledger["tracks"][track_name], "stage_a", track_root, stage_a_candidates, chosen_a["candidate_name"])
        _write_selection_ledger(ledger_path, ledger)

        stage_b_candidates = []
        for policy in suite["training_policies"]:
            candidate_name = policy["name"]
            experiment = _merge_dicts(chosen_a["experiment"], policy["overrides"])
            summary = _run_experiment(track, experiment, track_root / "stage_b" / candidate_name, "stage_b", candidate_name, suite["dev_seeds"], False, suite)
            stage_b_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
        chosen_b = _stage_choice(stage_b_candidates)
        _record_stage(ledger["tracks"][track_name], "stage_b", track_root, stage_b_candidates, chosen_b["candidate_name"])
        _write_selection_ledger(ledger_path, ledger)

        stage_c_candidates = []
        for capacity in suite["capacity_candidates"]:
            candidate_name = capacity["name"]
            experiment = _merge_dicts(chosen_b["experiment"], capacity["overrides"])
            summary = _run_experiment(track, experiment, track_root / "stage_c" / candidate_name, "stage_c", candidate_name, suite["dev_seeds"], False, suite)
            stage_c_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
        chosen_c = _stage_choice(stage_c_candidates)
        _record_stage(ledger["tracks"][track_name], "stage_c", track_root, stage_c_candidates, chosen_c["candidate_name"])
        _write_selection_ledger(ledger_path, ledger)

        if track["dataset"] == "dvlog":
            stage_d_candidates = []
            for normalization_source in suite["normalization_protocols"]["dvlog"]:
                candidate_name = f"norm_{normalization_source}"
                experiment = _merge_dicts(chosen_c["experiment"], {"normalization_source": normalization_source})
                summary = _run_experiment(track, experiment, track_root / "stage_d" / candidate_name, "stage_d", candidate_name, suite["dev_seeds"], False, suite)
                stage_d_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
            chosen_d = _stage_choice(stage_d_candidates)
            _record_stage(ledger["tracks"][track_name], "stage_d", track_root, stage_d_candidates, chosen_d["candidate_name"])
            final_choice = chosen_d
        else:
            chosen_c["experiment"]["normalization_source"] = "train"
            ledger["tracks"][track_name]["stages"]["stage_d"] = {
                "candidates": [
                    {
                        "candidate_name": "norm_train",
                        "summary_path": None,
                        "dev_macro_f1_mean": chosen_c["summary"]["dev_metrics_mean"]["macro_f1"],
                        "selected_aggregation": chosen_c["summary"]["selected_aggregation"],
                    }
                ],
                "selected": "norm_train",
                "note": "E-DAIC keeps train-only normalization fixed for this milestone.",
            }
            final_choice = chosen_c

        ledger["tracks"][track_name]["final_config"] = final_choice["experiment"]
        ledger["tracks"][track_name]["frozen_aggregation"] = final_choice["summary"]["selected_aggregation"]
        _write_selection_ledger(ledger_path, ledger)

    ledger["completed_at"] = _timestamp()
    _write_selection_ledger(ledger_path, ledger)
    _generate_leaderboard(output_root)


def _build_frozen_ledger(suite: dict[str, Any]) -> dict[str, Any]:
    frozen = {"suite_name": suite["suite_name"], "completed_at": _timestamp(), "tracks": {}}
    for track in suite["tracks"]:
        frozen_track = track.get("frozen_final") or suite.get("frozen_tracks", {}).get(track["name"])
        if frozen_track is None:
            raise RuntimeError(f"Missing frozen_final config for track {track['name']}.")
        if "source_selection_ledger" in frozen_track:
            source_ledger = _load_selection_ledger(frozen_track["source_selection_ledger"])
            source_track_name = frozen_track.get("source_track", track["name"])
            source_track = source_ledger.get("tracks", {}).get(source_track_name)
            if source_track is None:
                raise RuntimeError(
                    f"Track {source_track_name} not found in source selection ledger "
                    f"{frozen_track['source_selection_ledger']}."
                )
            resolved_frozen_track = {
                "final_config": source_track["final_config"],
                "frozen_aggregation": source_track["frozen_aggregation"],
            }
            if "overrides" in frozen_track:
                resolved_frozen_track["final_config"] = _merge_dicts(
                    resolved_frozen_track["final_config"],
                    frozen_track["overrides"],
                )
            frozen_track = resolved_frozen_track
        frozen["tracks"][track["name"]] = {
            "dataset": track["dataset"],
            "modality": track.get("bundle_name", track.get("modality")),
            "stages": {"frozen": {"candidates": [], "selected": "locked_run"}},
            "final_config": frozen_track["final_config"],
            "frozen_aggregation": frozen_track.get("frozen_aggregation", frozen_track["final_config"].get("subject_aggregation", "transformer")),
        }
    return frozen


def _run_dev_stage(suite: dict[str, Any], output_root: Path):
    if any("search_stages" in track for track in suite["tracks"]):
        _run_generic_dev_stage(suite, output_root)
    else:
        _run_classic_dev_stage(suite, output_root)


def _run_finalize_stage(suite: dict[str, Any], output_root: Path):
    ledger_path = output_root / "selection_ledger.json"
    if suite.get("frozen_tracks") or any("frozen_final" in track for track in suite["tracks"]):
        ledger = _build_frozen_ledger(suite)
        _write_selection_ledger(ledger_path, ledger)
    else:
        if not ledger_path.exists():
            raise RuntimeError("Cannot finalize without a completed dev-stage selection ledger.")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for track in suite["tracks"]:
        track_ledger = ledger["tracks"].get(track["name"])
        if not track_ledger or not track_ledger.get("final_config") or not track_ledger.get("frozen_aggregation"):
            raise RuntimeError(f"Track {track['name']} is missing a frozen final config.")
        _run_experiment(
            track=track,
            experiment=track_ledger["final_config"],
            experiment_dir=output_root / "final" / track["name"] / "locked_run",
            stage_name="final_locked",
            candidate_name="locked_run",
            seeds=suite["final_seeds"],
            include_test=True,
            suite=suite,
            forced_aggregation=track_ledger["frozen_aggregation"],
        )

    _generate_leaderboard(output_root)
    _write_final_report(output_root, ledger, suite)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark-quality experiment suites.")
    parser.add_argument("--suite", required=True, help="Path to JSON suite config.")
    parser.add_argument("--stage", choices=["dev", "finalize"], required=True)
    args = parser.parse_args()

    suite_path = _resolve_path(args.suite)
    suite = _load_suite(suite_path)
    output_root = _resolve_path(suite["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "dev":
        _run_dev_stage(suite, output_root)
    else:
        _run_finalize_stage(suite, output_root)


if __name__ == "__main__":
    main()
