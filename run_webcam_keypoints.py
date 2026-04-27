from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np
import torch

from train_keypoints_one_file import SimpleHandKeypointCNN


DEFAULT_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str | None = None
    score: float | None = None


@dataclass
class LetterboxMeta:
    src_w: int
    src_h: int
    pad_x: float
    pad_y: float
    scale_x: float
    scale_y: float
    bbox: BBox
    frame_w: int
    frame_h: int


@dataclass
class InferDebug:
    bbox_preview: np.ndarray | None
    letterbox_preview: np.ndarray | None
    final_preview: np.ndarray | None


class HandDetector:
    def __init__(self, backend: str, detector: Any, is_video: bool) -> None:
        self.backend = backend
        self.detector = detector
        self.is_video = is_video
        self._last_timestamp_ms = 0

    def _next_timestamp_ms(self) -> int:
        now = int(time.time() * 1000)
        if now <= self._last_timestamp_ms:
            now = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now
        return now

    def detect(self, frame_rgb: np.ndarray) -> tuple[list[Any], list[Any]]:
        if self.backend == "solutions":
            results = self.detector.process(frame_rgb)
            return (results.multi_hand_landmarks or [], results.multi_handedness or [])

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.is_video:
            ts = self._next_timestamp_ms()
            result = self.detector.detect_for_video(mp_image, ts)
        else:
            result = self.detector.detect(mp_image)
        return (getattr(result, "hand_landmarks", []) or [], getattr(result, "handedness", []) or [])

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run hand keypoint pipeline: "
            "hand detection -> bbox crop -> letterbox(224) -> model -> unletterbox+uncrop."
        )
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-keypoints", type=int, default=21)
    parser.add_argument("--max-num-hands", type=int, default=2)
    parser.add_argument("--bbox-margin", type=float, default=0.25)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--hand-landmarker-model",
        type=str,
        default=None,
        help="Path to hand_landmarker.task (MediaPipe tasks backend).",
    )
    parser.add_argument(
        "--image-paths",
        nargs="*",
        default=None,
        help="Optional image paths for offline inference. If set, webcam is not used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir for image mode. Default: ./inference_outputs",
    )
    parser.add_argument(
        "--save-pipeline-preview",
        action="store_true",
        default=True,
        help="Save pipeline preview image in image mode (default: true).",
    )
    parser.add_argument(
        "--no-save-pipeline-preview",
        dest="save_pipeline_preview",
        action="store_false",
    )
    parser.add_argument(
        "--normalized-output",
        action="store_true",
        default=True,
        help="Model outputs are normalized to [0,1] (default for this training script).",
    )
    parser.add_argument(
        "--no-normalized-output",
        dest="normalized_output",
        action="store_false",
        help="Use when model outputs pixel coordinates in resized image space.",
    )
    parser.add_argument("--no-flip", dest="flip", action="store_false", default=True)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args, _unknown = parser.parse_known_args()
    return args


def resolve_checkpoint_path(path_arg: str | None) -> Path:
    if path_arg:
        ckpt = Path(path_arg).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    script_dir = Path(__file__).resolve().parent
    search_roots: list[Path] = [Path.cwd().resolve(), script_dir, script_dir.parent.resolve()]
    seen: set[str] = set()
    all_pt_files: list[Path] = []
    best_named_files: list[Path] = []

    for root in search_roots:
        key = str(root)
        if key in seen or not root.exists():
            continue
        seen.add(key)
        for p in root.glob("*.pt"):
            if not p.is_file():
                continue
            all_pt_files.append(p.resolve())
            if p.name.startswith("best_keypoint_model"):
                best_named_files.append(p.resolve())

    candidates = best_named_files if best_named_files else all_pt_files
    if not candidates:
        checked = "\n".join(str(r) for r in search_roots)
        raise FileNotFoundError(
            "Checkpoint not found. Pass explicit path with --checkpoint.\n"
            f"Checked roots:\n{checked}"
        )

    # Use the most recently modified checkpoint.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(
    checkpoint_path: Path,
    image_size: int,
    num_keypoints: int,
    device: torch.device,
) -> tuple[SimpleHandKeypointCNN, int | None, float | None]:
    model = SimpleHandKeypointCNN(num_keypoints=num_keypoints).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_epoch: int | None = None
    checkpoint_val_loss: float | None = None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch_raw = checkpoint.get("epoch", None)
        if isinstance(epoch_raw, (int, np.integer)):
            checkpoint_epoch = int(epoch_raw)
        val_loss_raw = checkpoint.get("val_loss", None)
        if isinstance(val_loss_raw, (float, int, np.floating, np.integer)):
            checkpoint_val_loss = float(val_loss_raw)
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    with torch.no_grad():
        _ = model(torch.zeros(1, 3, image_size, image_size, device=device))

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint_epoch, checkpoint_val_loss


def handedness_info(multi_handedness: Sequence[Any], hand_idx: int) -> tuple[str | None, float | None]:
    if hand_idx >= len(multi_handedness):
        return None, None

    entry = multi_handedness[hand_idx]
    candidates: list[Any]
    if hasattr(entry, "classification"):
        candidates = list(entry.classification)
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
    if hasattr(landmarks, "landmark"):
        return landmarks.landmark
    return landmarks


def to_landmark_pixels(landmarks: Any, frame_w: int, frame_h: int) -> np.ndarray:
    pts: list[list[float]] = []
    for lm in iter_landmarks(landmarks):
        pts.append([float(lm.x) * frame_w, float(lm.y) * frame_h])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def compute_bbox_from_points(
    points_xy: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin: float,
) -> BBox | None:
    if points_xy.shape[0] == 0:
        return None
    x_min = float(points_xy[:, 0].min())
    y_min = float(points_xy[:, 1].min())
    x_max = float(points_xy[:, 0].max())
    y_max = float(points_xy[:, 1].max())

    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)
    x_min -= bw * margin
    y_min -= bh * margin
    x_max += bw * margin
    y_max += bh * margin

    x1 = int(max(0, np.floor(x_min)))
    y1 = int(max(0, np.floor(y_min)))
    x2 = int(min(frame_w, np.ceil(x_max)))
    y2 = int(min(frame_h, np.ceil(y_max)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def letterbox_bgr(image_bgr: np.ndarray, target_size: int) -> tuple[np.ndarray, float, float, float, float]:
    src_h, src_w = image_bgr.shape[:2]
    scale = min(target_size / float(src_w), target_size / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0
    left = int(np.floor(pad_x))
    top = int(np.floor(pad_y))
    canvas[top : top + new_h, left : left + new_w] = resized

    scale_x = new_w / float(src_w)
    scale_y = new_h / float(src_h)
    return canvas, pad_x, pad_y, scale_x, scale_y


def preprocess_hand_crops(
    frame_bgr: np.ndarray,
    bboxes: list[BBox],
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, list[LetterboxMeta], list[np.ndarray]]:
    tensors: list[torch.Tensor] = []
    metas: list[LetterboxMeta] = []
    letterboxed_images: list[np.ndarray] = []
    frame_h, frame_w = frame_bgr.shape[:2]

    for bbox in bboxes:
        crop = frame_bgr[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        if crop.size == 0:
            continue

        lb_bgr, pad_x, pad_y, scale_x, scale_y = letterbox_bgr(crop, image_size)
        lb_rgb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(lb_rgb).to(device=device, dtype=torch.float32)
        x = x.permute(2, 0, 1).contiguous() / 255.0

        tensors.append(x)
        letterboxed_images.append(lb_bgr)
        metas.append(
            LetterboxMeta(
                src_w=int(crop.shape[1]),
                src_h=int(crop.shape[0]),
                pad_x=float(pad_x),
                pad_y=float(pad_y),
                scale_x=float(scale_x),
                scale_y=float(scale_y),
                bbox=bbox,
                frame_w=frame_w,
                frame_h=frame_h,
            )
        )

    if not tensors:
        return None, [], []
    return torch.stack(tensors, dim=0), metas, letterboxed_images


def map_prediction_to_frame(
    pred_xy: np.ndarray,
    meta: LetterboxMeta,
    image_size: int,
    normalized_output: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pred = pred_xy.copy()
    if normalized_output:
        pred[:, 0] = np.clip(pred[:, 0], 0.0, 1.0) * float(image_size)
        pred[:, 1] = np.clip(pred[:, 1], 0.0, 1.0) * float(image_size)

    # unletterbox: remove padding and divide by resize scales
    pred_crop = pred.copy()
    pred_crop[:, 0] = (pred_crop[:, 0] - meta.pad_x) / max(meta.scale_x, 1e-8)
    pred_crop[:, 1] = (pred_crop[:, 1] - meta.pad_y) / max(meta.scale_y, 1e-8)
    pred_crop[:, 0] = np.clip(pred_crop[:, 0], 0.0, float(meta.src_w - 1))
    pred_crop[:, 1] = np.clip(pred_crop[:, 1], 0.0, float(meta.src_h - 1))

    # uncrop: add crop offset back to original frame coordinates
    pred_frame = pred_crop.copy()
    pred_frame[:, 0] += float(meta.bbox.x1)
    pred_frame[:, 1] += float(meta.bbox.y1)
    pred_frame[:, 0] = np.clip(pred_frame[:, 0], 0.0, float(meta.frame_w - 1))
    pred_frame[:, 1] = np.clip(pred_frame[:, 1], 0.0, float(meta.frame_h - 1))
    return pred, pred_frame


def draw_keypoints(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    point_radius: int,
    line_thickness: int,
) -> None:
    if keypoints_xy.shape[0] < 21:
        return

    for i, j in HAND_CONNECTIONS:
        pi = tuple(np.round(keypoints_xy[i]).astype(int))
        pj = tuple(np.round(keypoints_xy[j]).astype(int))
        cv2.line(frame, pi, pj, (0, 220, 120), line_thickness, lineType=cv2.LINE_AA)

    for k, (x, y) in enumerate(keypoints_xy):
        center = (int(round(float(x))), int(round(float(y))))
        color = (0, 140, 255) if k == 0 else (255, 120, 0)
        cv2.circle(frame, center, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, bbox: BBox, idx: int) -> None:
    cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 220, 40), 2, cv2.LINE_AA)
    text = f"hand {idx}"
    if bbox.label:
        text += f" {bbox.label}"
    if bbox.score is not None:
        text += f" {bbox.score:.2f}"
    cv2.putText(
        frame,
        text,
        (bbox.x1, max(18, bbox.y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 220, 40),
        2,
        cv2.LINE_AA,
    )


def create_hand_detector(
    args: argparse.Namespace,
    script_dir: Path,
    is_video: bool,
) -> HandDetector:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        detector = mp.solutions.hands.Hands(
            static_image_mode=(not is_video),
            max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        return HandDetector(backend="solutions", detector=detector, is_video=is_video)

    model_path = (
        Path(args.hand_landmarker_model).expanduser()
        if args.hand_landmarker_model
        else (script_dir / "hand_landmarker.task")
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"hand_landmarker.task not found: {model_path}\n"
            f"Download: {DEFAULT_HAND_LANDMARKER_URL}\n"
            "Or pass --hand-landmarker-model."
        )

    try:
        model_asset_path = str(model_path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        model_asset_path = str(model_path)

    tasks = mp.tasks
    vision = tasks.vision
    running_mode = vision.RunningMode.VIDEO if is_video else vision.RunningMode.IMAGE

    base_options = tasks.BaseOptions(model_asset_path=model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=args.max_num_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return HandDetector(backend="tasks", detector=detector, is_video=is_video)


def detect_bboxes(
    detector: HandDetector,
    frame_bgr: np.ndarray,
    bbox_margin: float,
) -> list[BBox]:
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_landmarks, multi_handedness = detector.detect(frame_rgb)

    bboxes: list[BBox] = []
    for hand_idx, landmarks in enumerate(hand_landmarks):
        pts = to_landmark_pixels(landmarks, frame_w=frame_w, frame_h=frame_h)
        bbox = compute_bbox_from_points(
            points_xy=pts,
            frame_w=frame_w,
            frame_h=frame_h,
            margin=bbox_margin,
        )
        if bbox is None:
            continue
        label, score = handedness_info(multi_handedness, hand_idx)
        bbox.label = label
        bbox.score = score
        bboxes.append(bbox)
    return bboxes


def run_inference_pipeline(
    frame_bgr: np.ndarray,
    model: SimpleHandKeypointCNN,
    detector: HandDetector,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, InferDebug, int]:
    annotated = frame_bgr.copy()
    bbox_preview = frame_bgr.copy()
    bboxes = detect_bboxes(detector=detector, frame_bgr=frame_bgr, bbox_margin=args.bbox_margin)
    if not bboxes:
        return annotated, InferDebug(bbox_preview=None, letterbox_preview=None, final_preview=None), 0

    for i, bbox in enumerate(bboxes, start=1):
        draw_bbox(bbox_preview, bbox, i)

    batch, metas, lb_imgs = preprocess_hand_crops(
        frame_bgr=frame_bgr,
        bboxes=bboxes,
        image_size=args.image_size,
        device=device,
    )
    if batch is None:
        return annotated, InferDebug(bbox_preview=None, letterbox_preview=None, final_preview=None), 0

    with torch.no_grad():
        pred_batch = model(batch).detach().cpu().numpy()

    first_letterbox: np.ndarray | None = None
    for idx, (pred_xy, meta, lb_img) in enumerate(zip(pred_batch, metas, lb_imgs), start=1):
        pred_lb, pred_frame = map_prediction_to_frame(
            pred_xy=pred_xy,
            meta=meta,
            image_size=args.image_size,
            normalized_output=args.normalized_output,
        )
        draw_bbox(annotated, meta.bbox, idx)
        draw_keypoints(
            frame=annotated,
            keypoints_xy=pred_frame,
            point_radius=args.point_radius,
            line_thickness=args.line_thickness,
        )

        if first_letterbox is None:
            preview = lb_img.copy()
            draw_keypoints(
                frame=preview,
                keypoints_xy=pred_lb,
                point_radius=max(1, args.point_radius - 1),
                line_thickness=max(1, args.line_thickness - 1),
            )
            first_letterbox = preview

    return (
        annotated,
        InferDebug(
            bbox_preview=bbox_preview,
            letterbox_preview=first_letterbox,
            final_preview=annotated.copy(),
        ),
        len(bboxes),
    )


def read_image_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(path: Path, image_bgr: np.ndarray, jpeg_quality: int = 95) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    new_w = max(1, int(round(w * (height / float(h)))))
    return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)


def make_pipeline_preview(
    bbox_img: np.ndarray,
    letterbox_img: np.ndarray,
    final_img: np.ndarray,
) -> np.ndarray:
    target_h = 360
    a = resize_to_height(bbox_img, target_h)
    b = resize_to_height(letterbox_img, target_h)
    c = resize_to_height(final_img, target_h)

    cv2.putText(a, "1) bbox + crop", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(b, "2) letterbox 224x224", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        c,
        "3) unletterbox + uncrop",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.hstack([a, b, c])


def run_image_mode(
    image_paths: list[Path],
    output_dir: Path,
    model: SimpleHandKeypointCNN,
    detector: HandDetector,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"image mode | output_dir: {output_dir}")
    for image_path in image_paths:
        frame = read_image_unicode(image_path)
        if frame is None:
            print(f"skip unreadable image: {image_path}")
            continue

        annotated, debug, num_hands = run_inference_pipeline(
            frame_bgr=frame,
            model=model,
            detector=detector,
            args=args,
            device=device,
        )

        out_main = output_dir / f"{image_path.stem}_keypoints.jpg"
        ok_main = write_image_unicode(out_main, annotated)
        if not ok_main:
            print(f"save failed: {out_main}")
            continue

        if args.save_pipeline_preview and num_hands > 0 and debug.bbox_preview is not None and debug.letterbox_preview is not None:
            preview = make_pipeline_preview(
                bbox_img=debug.bbox_preview,
                letterbox_img=debug.letterbox_preview,
                final_img=debug.final_preview if debug.final_preview is not None else annotated,
            )
            out_preview = output_dir / f"{image_path.stem}_pipeline.jpg"
            if not write_image_unicode(out_preview, preview):
                print(f"save failed: {out_preview}")
            else:
                print(f"saved: {out_main.name} + {out_preview.name} | hands={num_hands}")
        else:
            print(f"saved: {out_main.name} | hands={num_hands}")


def run_webcam_mode(
    model: SimpleHandKeypointCNN,
    detector: HandDetector,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with camera id {args.camera_id}")

    win_name = "Hand Keypoints Pipeline (q/Esc to quit)"
    print("running webcam inference...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            annotated, _debug, num_hands = run_inference_pipeline(
                frame_bgr=frame,
                model=model,
                detector=detector,
                args=args,
                device=device,
            )

            cv2.putText(
                annotated,
                f"device: {device} | hands: {num_hands}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "pipeline: bbox/crop -> letterbox -> model -> unletterbox/uncrop",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 255, 220),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def resolve_input_images(image_path_args: Sequence[str] | None) -> list[Path]:
    if not image_path_args:
        return []
    paths: list[Path] = []
    for item in image_path_args:
        p = Path(item).expanduser().resolve()
        if p.exists() and p.is_file():
            paths.append(p)
        else:
            print(f"skip missing image: {p}")
    return paths


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, checkpoint_epoch, checkpoint_val_loss = load_model(
        checkpoint_path=checkpoint_path,
        image_size=args.image_size,
        num_keypoints=args.num_keypoints,
        device=device,
    )

    script_dir = Path(__file__).resolve().parent
    image_paths = resolve_input_images(args.image_paths)
    is_video = len(image_paths) == 0
    detector = create_hand_detector(args=args, script_dir=script_dir, is_video=is_video)

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    if checkpoint_epoch is not None:
        if checkpoint_val_loss is not None:
            print(f"checkpoint_info: epoch={checkpoint_epoch} val_loss={checkpoint_val_loss:.6f}")
        else:
            print(f"checkpoint_info: epoch={checkpoint_epoch}")
    print(f"detector_backend: {detector.backend}")

    try:
        if image_paths:
            output_dir = (
                Path(args.output_dir).expanduser().resolve()
                if args.output_dir
                else (script_dir / "inference_outputs").resolve()
            )
            run_image_mode(
                image_paths=image_paths,
                output_dir=output_dir,
                model=model,
                detector=detector,
                args=args,
                device=device,
            )
        else:
            run_webcam_mode(
                model=model,
                detector=detector,
                args=args,
                device=device,
            )
    finally:
        detector.close()


if __name__ == "__main__":
    main()
