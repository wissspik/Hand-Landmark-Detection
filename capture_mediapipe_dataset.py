from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np


DEFAULT_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture webcam images and MediaPipe hand keypoints every N seconds."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory for saved frames. Default: ./images/train_my next to this script.",
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        default=None,
        help="JSONL annotations path. Default: ./images/train_my.annotations.jsonl",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=None,
        help="Meta JSON path. Default: ./images/train_my.meta.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Legacy mode: save to <output-dir>/images + annotations/meta in <output-dir>.",
    )
    parser.add_argument(
        "--hand-landmarker-model",
        type=str,
        default=None,
        help=(
            "Path to hand_landmarker.task (required for new MediaPipe tasks API). "
            "Default: ./hand_landmarker.task next to this script."
        ),
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--max-num-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Save frames even when no hands are detected.",
    )
    parser.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        default=True,
        help="Disable horizontal flip.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality [0..100].",
    )
    args, _unknown = parser.parse_known_args()
    return args


def resolve_output_paths(args: argparse.Namespace, script_dir: Path) -> tuple[Path, Path, Path]:
    if args.output_dir:
        base = Path(args.output_dir).expanduser().resolve()
        images_dir = (
            Path(args.images_dir).expanduser().resolve()
            if args.images_dir
            else (base / "images").resolve()
        )
        annotations_path = (
            Path(args.annotations_path).expanduser().resolve()
            if args.annotations_path
            else (base / "annotations.jsonl").resolve()
        )
        meta_path = (
            Path(args.meta_path).expanduser().resolve()
            if args.meta_path
            else (base / "meta.json").resolve()
        )
        return images_dir, annotations_path, meta_path

    images_dir = (
        Path(args.images_dir).expanduser().resolve()
        if args.images_dir
        else (script_dir / "images" / "train_my").resolve()
    )
    annotations_path = (
        Path(args.annotations_path).expanduser().resolve()
        if args.annotations_path
        else (script_dir / "images" / "train_my.annotations.jsonl").resolve()
    )
    meta_path = (
        Path(args.meta_path).expanduser().resolve()
        if args.meta_path
        else (script_dir / "images" / "train_my.meta.json").resolve()
    )
    return images_dir, annotations_path, meta_path


def handedness_info(multi_handedness: list[Any] | None, hand_idx: int) -> tuple[str | None, float | None]:
    if not multi_handedness or hand_idx >= len(multi_handedness):
        return None, None

    entry = multi_handedness[hand_idx]
    candidates: list[Any] = []

    # Old API: results.multi_handedness[idx].classification[0]
    if hasattr(entry, "classification"):
        candidates = list(entry.classification)
    # New tasks API: list[Category]
    elif isinstance(entry, (list, tuple)):
        candidates = list(entry)
    else:
        candidates = [entry]

    if not candidates:
        return None, None

    first = candidates[0]
    label = getattr(first, "label", None) or getattr(first, "category_name", None)
    score_obj = getattr(first, "score", None)
    score = float(score_obj) if score_obj is not None else None
    return label, score


def iter_landmarks(landmarks: Any) -> Iterable[Any]:
    # Old API: NormalizedLandmarkList with .landmark
    if hasattr(landmarks, "landmark"):
        return landmarks.landmark
    # New tasks API: sequence of landmarks
    return landmarks


def landmarks_to_dicts(
    landmarks: Any, frame_w: int, frame_h: int
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    norm_points: list[dict[str, float]] = []
    px_points: list[dict[str, float]] = []

    for lm in iter_landmarks(landmarks):
        x_norm = float(lm.x)
        y_norm = float(lm.y)
        z_norm = float(getattr(lm, "z", 0.0))

        x_px = x_norm * frame_w
        y_px = y_norm * frame_h

        norm_points.append({"x": x_norm, "y": y_norm, "z": z_norm})
        px_points.append({"x": x_px, "y": y_px})

    return norm_points, px_points


def draw_hand_skeleton(
    frame: Any,
    points_px: list[dict[str, float]],
    connections: Iterable[Any],
) -> None:
    for conn in connections:
        start = int(conn.start) if hasattr(conn, "start") else int(conn[0])
        end = int(conn.end) if hasattr(conn, "end") else int(conn[1])
        if start >= len(points_px) or end >= len(points_px):
            continue
        p1 = points_px[start]
        p2 = points_px[end]
        cv2.line(
            frame,
            (int(round(p1["x"])), int(round(p1["y"]))),
            (int(round(p2["x"])), int(round(p2["y"]))),
            (0, 220, 120),
            2,
            cv2.LINE_AA,
        )

    for i, p in enumerate(points_px):
        color = (0, 140, 255) if i == 0 else (255, 120, 0)
        cv2.circle(
            frame,
            (int(round(p["x"])), int(round(p["y"]))),
            3,
            color,
            -1,
            cv2.LINE_AA,
        )


def write_jpeg(frame_bgr: Any, path: Path, quality: int) -> bool:
    # cv2.imwrite can fail on Windows unicode paths; imencode+tofile is safer.
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def get_initial_saved_count(images_dir: Path) -> int:
    pattern = re.compile(r"^frame_(\d{6})\.jpg$", re.IGNORECASE)
    max_idx = -1
    for p in images_dir.glob("frame_*.jpg"):
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def create_hand_detector(
    args: argparse.Namespace,
    script_dir: Path,
) -> tuple[str, Any, Iterable[Any]]:
    # Old API branch (if available)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        return "solutions", hands, mp.solutions.hands.HAND_CONNECTIONS

    # New tasks API branch
    model_path = (
        Path(args.hand_landmarker_model).expanduser()
        if args.hand_landmarker_model
        else (script_dir / "hand_landmarker.task")
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"hand_landmarker.task not found: {model_path}\n"
            f"Download and place the file there, or pass --hand-landmarker-model.\n"
            f"Model URL: {DEFAULT_HAND_LANDMARKER_URL}"
        )

    # Prefer relative path (works more reliably on Windows paths with Cyrillic symbols).
    try:
        model_asset_path = str(model_path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        model_asset_path = str(model_path)

    tasks = mp.tasks
    vision = tasks.vision
    base_options = tasks.BaseOptions(model_asset_path=model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_num_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return "tasks", detector, vision.HandLandmarksConnections.HAND_CONNECTIONS


def main() -> None:
    args = parse_args()

    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")
    if not (0 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [0..100]")

    script_dir = Path(__file__).resolve().parent
    images_dir, annotations_path, meta_path = resolve_output_paths(args, script_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    backend, detector, connections = create_hand_detector(args, script_dir)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam camera_id={args.camera_id}")

    saved_count = get_initial_saved_count(images_dir)
    last_save_ts = 0.0

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "camera_id": args.camera_id,
        "interval_sec": args.interval_sec,
        "max_num_hands": args.max_num_hands,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "flip": args.flip,
        "save_empty": args.save_empty,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    window_name = "MediaPipe Capture (q/Esc: quit, s: save now)"
    print(f"backend: {backend}")
    print(f"images_dir: {images_dir}")
    print(f"annotations: {annotations_path}")
    print("running...")

    try:
        with annotations_path.open("a", encoding="utf-8") as f_ann:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                if args.flip:
                    frame = cv2.flip(frame, 1)

                # Save raw frame, draw only on preview copy.
                frame_for_save = frame.copy()
                frame_h, frame_w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hands_list: list[dict[str, Any]] = []

                if backend == "solutions":
                    results = detector.process(frame_rgb)
                    hand_landmarks = results.multi_hand_landmarks or []
                    multi_handedness = results.multi_handedness or []
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    timestamp_ms = int(time.time() * 1000)
                    result = detector.detect_for_video(mp_image, timestamp_ms)
                    hand_landmarks = getattr(result, "hand_landmarks", []) or []
                    multi_handedness = getattr(result, "handedness", []) or []

                for hand_idx, hand_lm in enumerate(hand_landmarks):
                    label, score = handedness_info(multi_handedness, hand_idx)
                    norm_points, px_points = landmarks_to_dicts(hand_lm, frame_w, frame_h)
                    hands_list.append(
                        {
                            "hand_index": hand_idx,
                            "label": label,
                            "score": score,
                            "landmarks_norm": norm_points,
                            "landmarks_px": px_points,
                        }
                    )
                    draw_hand_skeleton(frame, px_points, connections)

                has_hands = len(hands_list) > 0

                cv2.putText(
                    frame,
                    f"saved={saved_count} hands={len(hands_list)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "s: save now | q/esc: quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

                now = time.time()
                should_save_by_interval = (now - last_save_ts) >= args.interval_sec
                force_save = key == ord("s")
                can_save = has_hands or args.save_empty

                if (should_save_by_interval or force_save) and can_save:
                    img_name = f"frame_{saved_count:06d}.jpg"
                    img_path = images_dir / img_name
                    ok_write = write_jpeg(frame_for_save, img_path, args.jpeg_quality)
                    if ok_write:
                        record = {
                            "image": img_name,
                            "width": frame_w,
                            "height": frame_h,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "num_hands": len(hands_list),
                            "hands": hands_list,
                        }
                        f_ann.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_ann.flush()
                        saved_count += 1
                        last_save_ts = now
                        reason = "manual" if force_save else "timer"
                        print(f"saved: {img_name} | hands={len(hands_list)} | reason={reason}")
                    else:
                        print(f"save failed: {img_path}")
                elif force_save and not can_save:
                    print("manual save skipped: no hands detected (use --save-empty to store empty frames).")
    finally:
        if hasattr(detector, "close"):
            detector.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"done. total saved frames: {saved_count}")


if __name__ == "__main__":
    main()
