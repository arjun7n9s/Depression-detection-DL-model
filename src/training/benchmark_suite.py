"""Config-driven benchmark runner for unimodal research experiments."""

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
from src.model.encoders import SequenceBinaryClassifier
from src.paths import PROJECT_ROOT, repo_relative
from src.training.trainer import TrainConfig, persist_seed_artifacts, set_seed, train_one_seed

INPUT_DIMS = {
    ("dvlog", "acoustic"): 25,
    ("dvlog", "visual"): 136,
    ("edaic", "acoustic"): 23,
    ("edaic", "visual"): 49,
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


def _build_loaders(track: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    if track["dataset"] == "dvlog":
        loaders = create_dvlog_dataloaders(
            batch_size=experiment["batch_size"],
            window_size=experiment["window_size"],
            stride=experiment["stride"],
            num_workers=experiment.get("num_workers", 0),
            balanced_sampling=experiment["balanced_sampling"],
            normalization_source=experiment.get("normalization_source", "train"),
        )
        return {"train": loaders["train"], "dev": loaders["valid"], "test": loaders["test"]}

    return create_edaic_dataloaders(
        modality=track["modality"],
        batch_size=experiment["batch_size"],
        window_size=experiment["window_size"],
        stride=experiment["stride"],
        num_workers=experiment.get("num_workers", 0),
        balanced_sampling=experiment["balanced_sampling"],
        min_valid_ratio=experiment.get("min_valid_ratio", 0.5),
    )


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
    seeds: list[int],
    seed_results: list[dict[str, Any]],
    selected_method: str,
    include_test: bool,
) -> dict[str, Any]:
    dev_rows = [payload["dev_results"][selected_method]["metrics"] for payload in seed_results]
    dev_mean, dev_std = _numeric_summary(dev_rows)
    summary = {
        "suite_track": track["name"],
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
    forced_aggregation: str | None = None,
) -> dict[str, Any]:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    (experiment_dir / "config.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    seed_results = []
    for seed in seeds:
        set_seed(seed)
        loaders = _build_loaders(track, experiment)
        model = SequenceBinaryClassifier(
            input_dim=INPUT_DIMS[(track["dataset"], track["modality"])],
            hidden_dim=experiment["hidden_dim"],
            num_layers=experiment["num_layers"],
            dropout=experiment["dropout"],
        )
        result = train_one_seed(
            model=model,
            train_loader=loaders["train"],
            dev_loader=loaders["dev"],
            test_loader=loaders["test"] if include_test else None,
            modality=track["modality"],
            config=_build_train_config(experiment),
        )
        seed_results.append({"seed": seed, **result})

    selected_method = forced_aggregation or _select_aggregation(seed_results, list(experiment["aggregation_methods"]))
    summary = _experiment_summary(
        track=track,
        experiment=experiment,
        stage_name=stage_name,
        candidate_name=candidate_name,
        seeds=seeds,
        seed_results=seed_results,
        selected_method=selected_method,
        include_test=include_test,
    )

    for payload in seed_results:
        seed_dir = experiment_dir / f"seed_{payload['seed']}"
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


def _write_final_report(output_root: Path, ledger: dict[str, Any]):
    final_rows = []
    for track_name, track_ledger in ledger["tracks"].items():
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
                "loss_name": summary["experiment_config"]["loss_name"],
                "balanced_sampling": summary["experiment_config"]["balanced_sampling"],
                "use_pos_weight": summary["experiment_config"].get("use_pos_weight", False),
                "hidden_dim": summary["experiment_config"]["hidden_dim"],
                "num_layers": summary["experiment_config"]["num_layers"],
                "normalization_source": summary["experiment_config"].get("normalization_source", "train"),
                "dev_macro_f1_mean": summary["dev_metrics_mean"].get("macro_f1"),
                "dev_macro_f1_std": summary["dev_metrics_std"].get("macro_f1"),
                "test_macro_f1_mean": summary["test_metrics_mean"].get("macro_f1"),
                "test_macro_f1_std": summary["test_metrics_std"].get("macro_f1"),
            }
        )

    if not final_rows:
        return

    frame = pd.DataFrame(final_rows).sort_values(by=["dataset", "test_macro_f1_mean"], ascending=[True, False])
    frame.to_csv(output_root / "final" / "benchmark_summary.csv", index=False)

    headers = list(frame.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        table_lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")

    lines = [
        "# Unimodal Benchmark Report",
        "",
        f"- Generated: {_timestamp()}",
        f"- Suite: {ledger['suite_name']}",
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
            lines.append(
                f"- {row['modality']}: test macro F1 {row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f}"
            )
        lines.append("")

    dvlog_norms = frame[frame["dataset"] == "dvlog"]["normalization_source"].unique().tolist()
    norm_note = ", ".join(sorted(str(item) for item in dvlog_norms)) if dvlog_norms else "not finalized"
    lines.extend(
        [
            "## Conclusions",
            "",
            f"- D-Vlog normalization conclusion: final selected normalization sources = {norm_note}.",
            "- Bimodal work is justified only if the next model is required to beat the stronger unimodal result on each dataset.",
            "- This report is the source of truth for benchmark-quality unimodal performance.",
        ]
    )

    report_path = output_root / "final" / "milestone_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _base_experiment_config(suite: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    config = dict(suite["base_experiment"])
    config["aggregation_methods"] = list(config["aggregation_methods"])
    config["normalization_source"] = track.get("default_normalization", config.get("normalization_source", "train"))
    return config


def _run_dev_stage(suite: dict[str, Any], output_root: Path):
    ledger_path = output_root / "selection_ledger.json"
    ledger = {
        "suite_name": suite["suite_name"],
        "completed_at": None,
        "tracks": {},
    }

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
            experiment = _merge_dicts(
                base_config,
                {
                    "window_size": window_size,
                    "stride": _window_stride(window_size, suite["stride_policy"]),
                },
            )
            summary = _run_experiment(
                track=track,
                experiment=experiment,
                experiment_dir=track_root / "stage_a" / candidate_name,
                stage_name="stage_a",
                candidate_name=candidate_name,
                seeds=suite["dev_seeds"],
                include_test=False,
            )
            stage_a_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
        chosen_a = _stage_choice(stage_a_candidates)
        ledger["tracks"][track_name]["stages"]["stage_a"] = {
            "candidates": [
                {
                    "candidate_name": item["candidate_name"],
                    "summary_path": repo_relative(track_root / "stage_a" / item["candidate_name"] / "summary.json"),
                    "dev_macro_f1_mean": item["summary"]["dev_metrics_mean"]["macro_f1"],
                    "selected_aggregation": item["summary"]["selected_aggregation"],
                }
                for item in stage_a_candidates
            ],
            "selected": chosen_a["candidate_name"],
        }
        _write_selection_ledger(ledger_path, ledger)

        stage_b_candidates = []
        for policy in suite["training_policies"]:
            candidate_name = policy["name"]
            experiment = _merge_dicts(chosen_a["experiment"], policy["overrides"])
            summary = _run_experiment(
                track=track,
                experiment=experiment,
                experiment_dir=track_root / "stage_b" / candidate_name,
                stage_name="stage_b",
                candidate_name=candidate_name,
                seeds=suite["dev_seeds"],
                include_test=False,
            )
            stage_b_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
        chosen_b = _stage_choice(stage_b_candidates)
        ledger["tracks"][track_name]["stages"]["stage_b"] = {
            "candidates": [
                {
                    "candidate_name": item["candidate_name"],
                    "summary_path": repo_relative(track_root / "stage_b" / item["candidate_name"] / "summary.json"),
                    "dev_macro_f1_mean": item["summary"]["dev_metrics_mean"]["macro_f1"],
                    "selected_aggregation": item["summary"]["selected_aggregation"],
                }
                for item in stage_b_candidates
            ],
            "selected": chosen_b["candidate_name"],
        }
        _write_selection_ledger(ledger_path, ledger)

        stage_c_candidates = []
        for capacity in suite["capacity_candidates"]:
            candidate_name = capacity["name"]
            experiment = _merge_dicts(chosen_b["experiment"], capacity["overrides"])
            summary = _run_experiment(
                track=track,
                experiment=experiment,
                experiment_dir=track_root / "stage_c" / candidate_name,
                stage_name="stage_c",
                candidate_name=candidate_name,
                seeds=suite["dev_seeds"],
                include_test=False,
            )
            stage_c_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
        chosen_c = _stage_choice(stage_c_candidates)
        ledger["tracks"][track_name]["stages"]["stage_c"] = {
            "candidates": [
                {
                    "candidate_name": item["candidate_name"],
                    "summary_path": repo_relative(track_root / "stage_c" / item["candidate_name"] / "summary.json"),
                    "dev_macro_f1_mean": item["summary"]["dev_metrics_mean"]["macro_f1"],
                    "selected_aggregation": item["summary"]["selected_aggregation"],
                }
                for item in stage_c_candidates
            ],
            "selected": chosen_c["candidate_name"],
        }
        _write_selection_ledger(ledger_path, ledger)

        if track["dataset"] == "dvlog":
            stage_d_candidates = []
            for normalization_source in suite["normalization_protocols"]["dvlog"]:
                candidate_name = f"norm_{normalization_source}"
                experiment = _merge_dicts(chosen_c["experiment"], {"normalization_source": normalization_source})
                summary = _run_experiment(
                    track=track,
                    experiment=experiment,
                    experiment_dir=track_root / "stage_d" / candidate_name,
                    stage_name="stage_d",
                    candidate_name=candidate_name,
                    seeds=suite["dev_seeds"],
                    include_test=False,
                )
                stage_d_candidates.append({"candidate_name": candidate_name, "summary": summary, "experiment": experiment})
            chosen_d = _stage_choice(stage_d_candidates)
            ledger["tracks"][track_name]["stages"]["stage_d"] = {
                "candidates": [
                    {
                        "candidate_name": item["candidate_name"],
                        "summary_path": repo_relative(track_root / "stage_d" / item["candidate_name"] / "summary.json"),
                        "dev_macro_f1_mean": item["summary"]["dev_metrics_mean"]["macro_f1"],
                        "selected_aggregation": item["summary"]["selected_aggregation"],
                    }
                    for item in stage_d_candidates
                ],
                "selected": chosen_d["candidate_name"],
            }
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


def _run_finalize_stage(suite: dict[str, Any], output_root: Path):
    ledger_path = output_root / "selection_ledger.json"
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
            forced_aggregation=track_ledger["frozen_aggregation"],
        )

    _generate_leaderboard(output_root)
    _write_final_report(output_root, ledger)


def main():
    parser = argparse.ArgumentParser(description="Run the benchmark-quality unimodal suite.")
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
