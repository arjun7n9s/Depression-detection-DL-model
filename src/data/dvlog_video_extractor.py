"""Vision V3 raw-video feature extractor for D-Vlog."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.dataset_audit import DVLOG_LABELS_CSV, DVLOG_VIDEO_LINKS_CSV
from src.paths import DVLOG_VIDEOS_DIR, PROCESSED_ROOT


VISION_V3_VIDEO_ROOT = PROCESSED_ROOT / "dvlog_video_v3"

BODY_DIM = 33 * 4
HAND_DIM = 2 * 21 * 3
GAZE_BLINK_DIM = 8
COLOR_HIST_DIM = 48
RESNET18_DIM = 512

@dataclass
class ExtractionConfig:
    output_root: Path = VISION_V3_VIDEO_ROOT
    target_fps: float = 1.0
    max_subjects: int | None = None
    overwrite: bool = False
    face_embedding_backend: str = "color_histogram"
    model_complexity: int = 1


@dataclass
class SubjectVideoRecord:
    subject_id: int
    label: str
    fold: str
    video_key: str
    video_path: Path


def _load_subject_video_records() -> list[SubjectVideoRecord]:
    labels: dict[int, dict[str, str]] = {}
    with open(DVLOG_LABELS_CSV, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[int(row["index"])] = {"label": row["label"], "fold": row["fold"]}

    records = []
    with open(DVLOG_VIDEO_LINKS_CSV, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = int(row["video_id"])
            label_row = labels.get(subject_id)
            if label_row is None:
                continue
            video_key = row["key"]
            video_path = DVLOG_VIDEOS_DIR / f"{video_key}.mp4"
            if not video_path.exists():
                continue
            records.append(
                SubjectVideoRecord(
                    subject_id=subject_id,
                    label=label_row["label"],
                    fold=label_row["fold"],
                    video_key=video_key,
                    video_path=video_path,
                )
            )
    return sorted(records, key=lambda item: item.subject_id)


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-6:
        return 0.0
    return float(numerator / denominator)


def _color_histogram_embedding(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((COLOR_HIST_DIM,), dtype=np.float32)
    hist_features = []
    for channel in range(3):
        hist = cv2.calcHist([crop], [channel], None, [16], [0, 256]).flatten()
        hist = hist / max(hist.sum(), 1.0)
        hist_features.append(hist.astype(np.float32))
    return np.concatenate(hist_features, axis=0)


class TorchvisionFaceEmbedder:
    def __init__(self):
        from torchvision.models import ResNet18_Weights, resnet18

        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            self.transforms = weights.transforms()
        except Exception:
            model = resnet18(weights=None)
            self.transforms = None
        self.model = torch.nn.Sequential(*list(model.children())[:-1]).eval()

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((RESNET18_DIM,), dtype=np.float32)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if self.transforms is not None:
            tensor = self.transforms(image).unsqueeze(0)
        else:
            resized = image.resize((224, 224))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            array = np.transpose(array, (2, 0, 1))
            tensor = torch.from_numpy(array).unsqueeze(0)
        embedding = self.model(tensor).reshape(-1).cpu().numpy().astype(np.float32)
        return embedding


def _build_face_embedder(backend: str):
    if backend == "none":
        return None
    if backend == "color_histogram":
        return _color_histogram_embedding
    if backend == "torchvision_resnet18":
        return TorchvisionFaceEmbedder()
    raise ValueError(f"Unsupported face embedding backend: {backend}")


def _expand_features(values: np.ndarray, target_dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros((target_dim,), dtype=np.float32)
    repeats = int(np.ceil(target_dim / values.size))
    expanded = np.tile(values, repeats)[:target_dim]
    return expanded.astype(np.float32)


def _build_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None
    return detector


def _center_bbox(frame_bgr: np.ndarray, fraction: float = 0.35) -> tuple[int, int, int, int]:
    height, width = frame_bgr.shape[:2]
    box_w = max(1, int(width * fraction))
    box_h = max(1, int(height * fraction))
    x0 = max(0, (width - box_w) // 2)
    y0 = max(0, int(height * 0.18))
    x1 = min(width, x0 + box_w)
    y1 = min(height, y0 + box_h)
    return x0, y0, x1, y1


def _detect_face_bbox(frame_bgr: np.ndarray, detector, previous_bbox: tuple[int, int, int, int] | None) -> tuple[tuple[int, int, int, int], bool]:
    if detector is None:
        if previous_bbox is not None:
            return previous_bbox, False
        return _center_bbox(frame_bgr), False

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        if previous_bbox is not None:
            return previous_bbox, False
        return _center_bbox(frame_bgr), False

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return (int(x), int(y), int(x + w), int(y + h)), True


def _crop_mean_std(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def _motion_summary(gray: np.ndarray, previous_gray: np.ndarray | None, bbox: tuple[int, int, int, int] | None = None) -> tuple[float, float, float]:
    if previous_gray is None:
        return 0.0, 0.0, 0.0
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        current = gray[y0:y1, x0:x1]
        previous = previous_gray[y0:y1, x0:x1]
    else:
        current = gray
        previous = previous_gray
    if current.size == 0 or previous.size == 0:
        return 0.0, 0.0, 0.0
    delta = cv2.absdiff(current, previous).astype(np.float32)
    return float(delta.mean()), float(delta.std()), float(delta.max() / 255.0)


def _extract_body_proxy(frame_bgr: np.ndarray, gray: np.ndarray, previous_gray: np.ndarray | None) -> np.ndarray:
    upper = frame_bgr[: max(1, frame_bgr.shape[0] * 3 // 4)]
    upper_gray = gray[: max(1, gray.shape[0] * 3 // 4)]
    prev_upper = previous_gray[: upper_gray.shape[0]] if previous_gray is not None else None
    rgb_mean = upper.mean(axis=(0, 1), dtype=np.float64) / 255.0
    rgb_std = upper.std(axis=(0, 1), dtype=np.float64) / 255.0
    motion_mean, motion_std, motion_peak = _motion_summary(upper_gray, prev_upper)
    edges = cv2.Canny(upper_gray, 80, 160).astype(np.float32) / 255.0
    edge_density = float(edges.mean())
    base = np.asarray(
        [
            *rgb_mean.tolist(),
            *rgb_std.tolist(),
            float(upper_gray.mean() / 255.0),
            float(upper_gray.std() / 255.0),
            motion_mean / 255.0,
            motion_std / 255.0,
            motion_peak,
            edge_density,
        ],
        dtype=np.float32,
    )
    return _expand_features(base, BODY_DIM)


def _extract_hand_proxy(gray: np.ndarray, previous_gray: np.ndarray | None) -> np.ndarray:
    height, width = gray.shape[:2]
    upper_half = gray[: max(1, height // 2)]
    left_region = upper_half[:, : max(1, width // 3)]
    right_region = upper_half[:, max(1, width * 2 // 3):]
    prev_left = previous_gray[: left_region.shape[0], : left_region.shape[1]] if previous_gray is not None else None
    prev_right = previous_gray[: right_region.shape[0], width - right_region.shape[1]:] if previous_gray is not None else None
    left_motion = _motion_summary(left_region, prev_left)
    right_motion = _motion_summary(right_region, prev_right)
    left_stats = [float(left_region.mean() / 255.0), float(left_region.std() / 255.0)]
    right_stats = [float(right_region.mean() / 255.0), float(right_region.std() / 255.0)]
    base = np.asarray([*left_stats, *right_stats, *left_motion, *right_motion], dtype=np.float32)
    return _expand_features(base, HAND_DIM)


def _extract_gaze_blink_proxy(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    previous_gray: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    detected_face: bool,
) -> np.ndarray:
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = bbox
    face_mean, face_std = _crop_mean_std(frame_bgr, bbox)
    face_motion = _motion_summary(gray, previous_gray, bbox)
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    features = np.asarray(
        [
            float(detected_face),
            float((x0 + x1) * 0.5 / max(width, 1)),
            float((y0 + y1) * 0.5 / max(height, 1)),
            float(bbox_w / max(width, 1)),
            float(bbox_h / max(height, 1)),
            face_mean / 255.0,
            face_std / 255.0,
            face_motion[0] / 255.0,
        ],
        dtype=np.float32,
    )
    return features


def _sample_step(source_fps: float, target_fps: float) -> int:
    if source_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def _subject_output_dir(root: Path, subject_id: int) -> Path:
    return root / str(subject_id)


def _subject_is_complete(output_dir: Path, face_backend: str) -> bool:
    required = [
        output_dir / "body_pose.npy",
        output_dir / "body_pose_mask.npy",
        output_dir / "hand_pose.npy",
        output_dir / "hand_pose_mask.npy",
        output_dir / "gaze_blink.npy",
        output_dir / "gaze_blink_mask.npy",
        output_dir / "frame_timestamps.npy",
        output_dir / "metadata.json",
    ]
    if face_backend != "none":
        required.extend(
            [
                output_dir / "face_affect_embed.npy",
                output_dir / "face_affect_embed_mask.npy",
            ]
        )
    return all(path.exists() for path in required)


def extract_subject_video(record: SubjectVideoRecord, config: ExtractionConfig, face_embedder) -> dict[str, Any]:
    output_dir = _subject_output_dir(config.output_root, record.subject_id)
    if output_dir.exists() and _subject_is_complete(output_dir, config.face_embedding_backend) and not config.overwrite:
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        return {"subject_id": record.subject_id, "status": "skipped", "metadata": metadata}

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(record.video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = _sample_step(fps, config.target_fps)
    face_detector = _build_face_detector()

    timestamps = []
    body_rows = []
    body_mask = []
    hand_rows = []
    hand_mask = []
    gaze_rows = []
    gaze_mask = []
    face_rows = []
    face_mask = []

    frame_index = 0
    previous_gray = None
    previous_face_bbox: tuple[int, int, int, int] | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox, detected_face = _detect_face_bbox(frame, face_detector, previous_face_bbox)
        previous_face_bbox = bbox

        body_values = _extract_body_proxy(frame, gray, previous_gray)
        hand_values = _extract_hand_proxy(gray, previous_gray)
        gaze_values = _extract_gaze_blink_proxy(frame, gray, previous_gray, bbox, detected_face)

        body_rows.append(body_values)
        body_mask.append(True)
        hand_rows.append(hand_values)
        hand_mask.append(True)
        gaze_rows.append(gaze_values)
        gaze_mask.append(True)

        if face_embedder is not None:
            has_face = bbox[2] > bbox[0] and bbox[3] > bbox[1]
            if has_face:
                embedding = face_embedder(frame, bbox)
            else:
                embedding = np.zeros((RESNET18_DIM if config.face_embedding_backend == "torchvision_resnet18" else COLOR_HIST_DIM,), dtype=np.float32)
            face_rows.append(embedding)
            face_mask.append(bool(detected_face))

        timestamps.append(float(frame_index / fps) if fps > 0 else float(len(timestamps)))
        previous_gray = gray
        frame_index += 1

    capture.release()

    body_arr = np.stack(body_rows, axis=0) if body_rows else np.zeros((0, BODY_DIM), dtype=np.float32)
    hand_arr = np.stack(hand_rows, axis=0) if hand_rows else np.zeros((0, HAND_DIM), dtype=np.float32)
    gaze_arr = np.stack(gaze_rows, axis=0) if gaze_rows else np.zeros((0, GAZE_BLINK_DIM), dtype=np.float32)
    np.save(output_dir / "body_pose.npy", body_arr.astype(np.float32))
    np.save(output_dir / "body_pose_mask.npy", np.asarray(body_mask, dtype=bool))
    np.save(output_dir / "hand_pose.npy", hand_arr.astype(np.float32))
    np.save(output_dir / "hand_pose_mask.npy", np.asarray(hand_mask, dtype=bool))
    np.save(output_dir / "gaze_blink.npy", gaze_arr.astype(np.float32))
    np.save(output_dir / "gaze_blink_mask.npy", np.asarray(gaze_mask, dtype=bool))
    np.save(output_dir / "frame_timestamps.npy", np.asarray(timestamps, dtype=np.float32))

    face_dim = 0
    if face_embedder is not None:
        face_arr = np.stack(face_rows, axis=0) if face_rows else np.zeros((0, COLOR_HIST_DIM), dtype=np.float32)
        face_dim = int(face_arr.shape[1]) if face_arr.ndim == 2 else 0
        np.save(output_dir / "face_affect_embed.npy", face_arr.astype(np.float32))
        np.save(output_dir / "face_affect_embed_mask.npy", np.asarray(face_mask, dtype=bool))

    metadata = {
        "subject_id": record.subject_id,
        "label": record.label,
        "fold": record.fold,
        "video_key": record.video_key,
        "video_path": str(record.video_path),
        "frames_total": total_frames,
        "fps": fps,
        "target_fps": config.target_fps,
        "sampled_frames": int(len(timestamps)),
        "body_pose_shape": list(body_arr.shape),
        "hand_pose_shape": list(hand_arr.shape),
        "gaze_blink_shape": list(gaze_arr.shape),
        "body_valid_ratio": float(np.mean(body_mask)) if body_mask else 0.0,
        "hand_valid_ratio": float(np.mean(hand_mask)) if hand_mask else 0.0,
        "gaze_valid_ratio": float(np.mean(gaze_mask)) if gaze_mask else 0.0,
        "face_embedding_backend": config.face_embedding_backend,
        "visual_backend": "opencv_fallback",
        "face_affect_dim": face_dim,
        "face_valid_ratio": float(np.mean(face_mask)) if face_mask else 0.0,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"subject_id": record.subject_id, "status": "complete", "metadata": metadata}


def run_extraction(config: ExtractionConfig) -> dict[str, Any]:
    records = _load_subject_video_records()
    if config.max_subjects is not None:
        records = records[: config.max_subjects]
    config.output_root.mkdir(parents=True, exist_ok=True)

    face_embedder = _build_face_embedder(config.face_embedding_backend)

    report = {
        "extractor": "dvlog_video_extractor",
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
        },
        "videos_dir": str(DVLOG_VIDEOS_DIR),
        "subjects_found": len(records),
        "processed": [],
    }

    counts = {"complete": 0, "skipped": 0, "failed": 0}
    for index, record in enumerate(records, start=1):
        try:
            result = extract_subject_video(record, config, face_embedder)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            report["processed"].append(result)
            if index % 25 == 0 or index == len(records):
                print(
                    f"[{index}/{len(records)}] complete={counts.get('complete', 0)} "
                    f"skipped={counts.get('skipped', 0)} failed={counts.get('failed', 0)}"
                )
        except Exception as exc:
            counts["failed"] += 1
            report["processed"].append({"subject_id": record.subject_id, "status": "failed", "error": str(exc)})

    report["summary"] = counts
    report_path = config.output_root / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Vision V3 raw-video features for D-Vlog.")
    parser.add_argument("--output-root", type=Path, default=VISION_V3_VIDEO_ROOT)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--face-embedding-backend",
        choices=["none", "color_histogram", "torchvision_resnet18"],
        default="color_histogram",
    )
    parser.add_argument("--model-complexity", type=int, default=1)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    config = ExtractionConfig(
        output_root=args.output_root,
        target_fps=args.target_fps,
        max_subjects=args.max_subjects,
        overwrite=args.overwrite,
        face_embedding_backend=args.face_embedding_backend,
        model_complexity=args.model_complexity,
    )
    report = run_extraction(config)
    print("=" * 60)
    print("D-VLOG VISION V3 VIDEO EXTRACTION")
    print("=" * 60)
    print(f"Subjects found: {report['subjects_found']}")
    print(f"Output root: {config.output_root}")
    print(f"Summary: {report['summary']}")
    print(f"Report: {config.output_root / 'extraction_report.json'}")


if __name__ == "__main__":
    main()
