from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class HandKeypointDataset(Dataset):
    """
    Generic dataset over pre-collected hand keypoint samples.
    Each sample should contain:
      image_path, image_id, file_name, keypoints(63), width, height
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: int = 224,
        normalize_keypoints: bool = True,
        crop_margin: float = 0.25,
    ) -> None:
        self.samples = samples
        self.image_size = int(image_size)
        self.normalize_keypoints = normalize_keypoints
        self.crop_margin = float(crop_margin)

        if not self.samples:
            raise RuntimeError("No valid samples found for dataset.")
        if self.image_size <= 0:
            raise ValueError("image_size must be > 0")
        if self.crop_margin < 0:
            raise ValueError("crop_margin must be >= 0")

    def _compute_crop_box(
        self,
        keypoints_xy: np.ndarray,
        visibility: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        vis_mask = visibility > 0
        if bool(vis_mask.any()):
            pts = keypoints_xy[vis_mask]
        else:
            pts = keypoints_xy

        x_min = float(np.min(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        x_max = float(np.max(pts[:, 0]))
        y_max = float(np.max(pts[:, 1]))

        bw = max(1.0, x_max - x_min)
        bh = max(1.0, y_max - y_min)
        x_min -= bw * self.crop_margin
        y_min -= bh * self.crop_margin
        x_max += bw * self.crop_margin
        y_max += bh * self.crop_margin

        x1 = int(max(0, np.floor(x_min)))
        y1 = int(max(0, np.floor(y_min)))
        x2 = int(min(width, np.ceil(x_max)))
        y2 = int(min(height, np.ceil(y_max)))

        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return x1, y1, x2, y2

    def _letterbox_image_and_keypoints(
        self,
        image: Image.Image,
        keypoints_xy: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        crop_w, crop_h = image.size
        scale = min(self.image_size / float(crop_w), self.image_size / float(crop_h))
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))

        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size), color=(114, 114, 114))
        left = int((self.image_size - new_w) // 2)
        top = int((self.image_size - new_h) // 2)
        canvas.paste(resized, (left, top))

        scale_x = new_w / float(crop_w)
        scale_y = new_h / float(crop_h)
        pad_x = (self.image_size - new_w) / 2.0
        pad_y = (self.image_size - new_h) / 2.0

        out_xy = keypoints_xy.copy()
        out_xy[:, 0] = out_xy[:, 0] * scale_x + pad_x
        out_xy[:, 1] = out_xy[:, 1] * scale_y + pad_y
        out_xy[:, 0] = np.clip(out_xy[:, 0], 0.0, float(self.image_size - 1))
        out_xy[:, 1] = np.clip(out_xy[:, 1], 0.0, float(self.image_size - 1))
        return canvas, out_xy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        orig_w, orig_h = image.size

        kp = np.asarray(sample["keypoints"], dtype=np.float32).reshape(21, 3)
        keypoints_xy = kp[:, :2].copy()
        visibility_np = kp[:, 2].copy()

        x1, y1, x2, y2 = self._compute_crop_box(
            keypoints_xy=keypoints_xy,
            visibility=visibility_np,
            width=orig_w,
            height=orig_h,
        )

        crop = image.crop((x1, y1, x2, y2))
        keypoints_xy[:, 0] -= float(x1)
        keypoints_xy[:, 1] -= float(y1)

        image_lb, keypoints_xy = self._letterbox_image_and_keypoints(
            image=crop,
            keypoints_xy=keypoints_xy,
        )

        image_np = np.asarray(image_lb, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        keypoints_xy_t = torch.from_numpy(keypoints_xy).to(dtype=torch.float32)
        visibility = torch.from_numpy(visibility_np).to(dtype=torch.float32)

        if self.normalize_keypoints:
            keypoints_xy_t[:, 0] /= float(self.image_size)
            keypoints_xy_t[:, 1] /= float(self.image_size)

        return {
            "image": image_tensor,
            "keypoints": keypoints_xy_t,
            "visibility": visibility,
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
        }


def load_coco_split_samples(dataset_root: Path, split: str) -> List[Dict[str, Any]]:
    images_dir = dataset_root / "images" / split
    annotations_path = dataset_root / "coco_annotation" / split / "_annotations.coco.json"
    if not images_dir.exists() or not annotations_path.exists():
        return []

    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_by_image_id: Dict[int, Dict[str, Any]] = {}
    for ann in coco.get("annotations", []):
        kp = ann.get("keypoints", [])
        if len(kp) != 63:
            continue
        image_id = int(ann["image_id"])
        if image_id not in ann_by_image_id:
            ann_by_image_id[image_id] = ann

    samples: List[Dict[str, Any]] = []
    for img in coco.get("images", []):
        image_id = int(img["id"])
        ann = ann_by_image_id.get(image_id)
        if ann is None:
            continue
        image_path = images_dir / str(img["file_name"])
        if not image_path.exists():
            continue
        samples.append(
            {
                "image_path": image_path,
                "image_id": image_id,
                "file_name": str(img["file_name"]),
                "width": int(img["width"]),
                "height": int(img["height"]),
                "keypoints": [float(x) for x in ann["keypoints"]],
                "source": f"coco_{split}",
            }
        )
    return samples


def _read_image_gray_stats(path: Path) -> tuple[float, float]:
    try:
        image = Image.open(path).convert("L")
    except Exception:
        return 0.0, 0.0
    arr = np.asarray(image, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def _hand_to_keypoints_pixels(hand: Dict[str, Any], width: int, height: int) -> List[float] | None:
    landmarks_px = hand.get("landmarks_px", [])
    landmarks_norm = hand.get("landmarks_norm", [])

    points: List[tuple[float, float, float]] = []

    if isinstance(landmarks_px, list) and len(landmarks_px) == 21:
        for p in landmarks_px:
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            v = 2.0 if (0.0 <= x < width and 0.0 <= y < height) else 0.0
            points.append((x, y, v))
    elif isinstance(landmarks_norm, list) and len(landmarks_norm) == 21:
        for p in landmarks_norm:
            x = float(p.get("x", 0.0)) * width
            y = float(p.get("y", 0.0)) * height
            v = 2.0 if (0.0 <= x < width and 0.0 <= y < height) else 0.0
            points.append((x, y, v))
    else:
        return None

    if len(points) != 21:
        return None
    out: List[float] = []
    for x, y, v in points:
        out.extend([x, y, v])
    return out


def load_frame_records_from_jsonl(
    images_dir: Path,
    ann_path: Path,
) -> List[Dict[str, Any]]:
    if not images_dir.exists() or not ann_path.exists():
        return []

    latest_by_image: Dict[str, Dict[str, Any]] = {}
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_name = str(rec.get("image", ""))
            if not image_name:
                continue
            latest_by_image[image_name] = rec

    frames: List[Dict[str, Any]] = []
    for image_name, rec in sorted(latest_by_image.items(), key=lambda x: x[0]):
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        width = int(rec.get("width", 0))
        height = int(rec.get("height", 0))
        if width <= 0 or height <= 0:
            try:
                im = Image.open(image_path)
                width, height = int(im.size[0]), int(im.size[1])
            except Exception:
                continue

        hands = rec.get("hands", [])
        hand_samples: List[Dict[str, Any]] = []
        for hand_idx, hand in enumerate(hands):
            kps = _hand_to_keypoints_pixels(hand, width, height)
            if kps is None:
                continue
            hand_samples.append(
                {
                    "keypoints": kps,
                    "hand_index": int(hand.get("hand_index", hand_idx)),
                    "label": hand.get("label"),
                    "score": hand.get("score"),
                }
            )
        if not hand_samples:
            continue

        gray_mean, gray_std = _read_image_gray_stats(image_path)
        all_xy = np.array([h["keypoints"] for h in hand_samples], dtype=np.float32).reshape(-1, 21, 3)[:, :, :2]
        xs = all_xy[:, :, 0]
        ys = all_xy[:, :, 1]
        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = float(ys.min())
        y_max = float(ys.max())
        bbox_area_ratio = max(0.0, (x_max - x_min) * (y_max - y_min) / max(1.0, width * height))
        cx = ((x_min + x_max) * 0.5) / max(1.0, width)
        cy = ((y_min + y_max) * 0.5) / max(1.0, height)

        frames.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "width": width,
                "height": height,
                "hands": hand_samples,
                "feature": np.array(
                    [
                        float(len(hand_samples)),
                        float(gray_mean),
                        float(gray_std),
                        float(bbox_area_ratio),
                        float(cx),
                        float(cy),
                    ],
                    dtype=np.float32,
                ),
            }
        )
    return frames


def load_train_my_frame_records(dataset_root: Path) -> List[Dict[str, Any]]:
    return load_frame_records_from_jsonl(
        images_dir=dataset_root / "images" / "train_my",
        ann_path=dataset_root / "images" / "train_my.annotations.jsonl",
    )


def load_val_my_frame_records(dataset_root: Path) -> List[Dict[str, Any]]:
    return load_frame_records_from_jsonl(
        images_dir=dataset_root / "images" / "val_my",
        ann_path=dataset_root / "images" / "val_my.annotations.jsonl",
    )


def split_train_my_frames_by_outlier(
    frames: List[Dict[str, Any]],
    val_ratio: float = 0.15,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not frames:
        return [], []
    if val_ratio <= 0:
        return frames, []

    feats = np.stack([f["feature"] for f in frames], axis=0)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z = (feats - mean) / std
    scores = np.linalg.norm(z, axis=1)

    n = len(frames)
    k = int(round(n * val_ratio))
    k = min(max(k, 1), n)
    val_idx = set(np.argsort(scores)[-k:].tolist())

    train_frames: List[Dict[str, Any]] = []
    val_frames: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        if i in val_idx:
            val_frames.append(fr)
        else:
            train_frames.append(fr)
    return train_frames, val_frames


def frames_to_hand_samples(
    frames: List[Dict[str, Any]],
    start_image_id: int,
) -> tuple[List[Dict[str, Any]], int]:
    samples: List[Dict[str, Any]] = []
    next_image_id = start_image_id
    for fr in frames:
        for hand in fr["hands"]:
            samples.append(
                {
                    "image_path": fr["image_path"],
                    "image_id": next_image_id,
                    "file_name": f"{fr['image_name']}#hand{hand['hand_index']}",
                    "width": fr["width"],
                    "height": fr["height"],
                    "keypoints": hand["keypoints"],
                    "source": "train_my",
                }
            )
            next_image_id += 1
    return samples, next_image_id


class SimpleHandKeypointCNN(nn.Module):
    def __init__(self, num_keypoints: int = 21) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.view(-1, self.num_keypoints, 2)

def is_valid_dataset_root(path: Path) -> bool:
    checks = [
        path / "images" / "train_my",
        path / "images" / "val_my",
        path / "images" / "train_my.annotations.jsonl",
        path / "images" / "val_my.annotations.jsonl",
    ]
    return all(p.exists() for p in checks)


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    if dataset_root is not None:
        explicit = Path(dataset_root).expanduser().resolve()
        if not is_valid_dataset_root(explicit):
            raise FileNotFoundError(
                f"Invalid dataset root: {explicit}. "
                "Expected folders/files: "
                "images/train_my, images/val_my, "
                "images/train_my.annotations.jsonl, images/val_my.annotations.jsonl"
            )
        return explicit

    candidates: List[Path] = []

    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)

    cwd = Path.cwd().resolve()
    candidates.extend(
        [
            cwd,
            cwd / "sample_data",
            cwd / "hand_keypoint_dataset_26k",
            cwd / "hand_keypoint_dataset_26k" / "hand_keypoint_dataset_26k",
            Path("/content"),
            Path("/content/sample_data"),
        ]
    )

    # Also check direct children of common working roots in Colab.
    for parent in [cwd, Path("/content")]:
        if parent.exists():
            for child in parent.iterdir():
                if child.is_dir():
                    candidates.append(child)

    checked: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)

        if is_valid_dataset_root(candidate):
            return candidate

    checked_str = "\n".join(str(p) for p in checked)
    raise FileNotFoundError(
        "Could not auto-detect dataset root.\n"
        "Checked paths:\n"
        f"{checked_str}\n\n"
        "Pass explicit path with --dataset-root."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train simple hand keypoint model (script + Colab friendly)."
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument(
        "--normalize-keypoints",
        dest="normalize_keypoints",
        action="store_true",
    )
    parser.add_argument(
        "--no-normalize-keypoints",
        dest="normalize_keypoints",
        action="store_false",
    )
    parser.set_defaults(normalize_keypoints=True)

    # In Jupyter/Colab kernels, extra args like "-f ...kernel.json" can appear.
    args, _unknown = parser.parse_known_args()
    return args


def batch_visible_metric_sums(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    visibility: torch.Tensor,
) -> tuple[float, float, float, int]:
    visible_mask = visibility > 0
    visible_points = int(visible_mask.sum().item())
    if visible_points == 0:
        return 0.0, 0.0, 0.0, 0

    diff_visible = (pred_xy - gt_xy)[visible_mask]
    euclid = torch.sqrt((diff_visible**2).sum(dim=-1) + 1e-8)

    mme_sum = euclid.sum().item()
    mae_sum = diff_visible.abs().sum().item()
    mse_sum = (diff_visible**2).sum().item()
    return mme_sum, mae_sum, mse_sum, visible_points


def save_metrics_and_plots(history: Dict[str, List[float]], save_path: Path) -> None:
    metrics_path = save_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"saved metrics -> {metrics_path}")

    if not history["epoch"]:
        print("skip plot: no epochs were run.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"skip plot: matplotlib is not available ({e})")
        return

    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("SmoothL1Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_mme"], marker="o", label="train")
    axes[1].plot(epochs, history["val_mme"], marker="o", label="val")
    axes[1].set_title("MME")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Euclidean Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_mae"], marker="o", label="train_mae")
    axes[2].plot(epochs, history["val_mae"], marker="o", label="val_mae")
    axes[2].plot(epochs, history["train_rmse"], marker="o", label="train_rmse")
    axes[2].plot(epochs, history["val_rmse"], marker="o", label="val_rmse")
    axes[2].set_title("MAE / RMSE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = save_path.with_suffix(".metrics.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {plot_path}")


if __name__ == "__main__":
    args = parse_args()

    DATASET_ROOT = resolve_dataset_root(args.dataset_root)
    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    CROP_MARGIN = args.crop_margin
    NUM_WORKERS = args.num_workers
    LR_REDUCE_FACTOR = args.lr_reduce_factor
    LR_PLATEAU_PATIENCE = args.lr_plateau_patience
    MIN_LR = args.min_lr
    EARLY_STOP_PATIENCE = args.early_stop_patience
    NORMALIZE_KEYPOINTS = args.normalize_keypoints
    SAVE_PATH = (
        Path(args.save_path).expanduser().resolve()
        if args.save_path
        else Path.cwd() / "best_keypoint_model.pt"
    )
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CROP_MARGIN < 0:
        raise ValueError("--crop-margin must be >= 0")

    train_my_frames = load_train_my_frame_records(DATASET_ROOT)
    val_my_frames = load_val_my_frame_records(DATASET_ROOT)

    if not train_my_frames:
        raise RuntimeError(
            "train_my is empty. Expected annotated data in images/train_my and images/train_my.annotations.jsonl"
        )
    if not val_my_frames:
        raise RuntimeError(
            "val_my is empty. Expected annotated data in images/val_my and images/val_my.annotations.jsonl"
        )

    # Prevent leakage: do not train on frames that are present in val_my.
    val_names = {fr["image_name"] for fr in val_my_frames}
    train_my_train_frames = [fr for fr in train_my_frames if fr["image_name"] not in val_names]
    train_my_val_frames = val_my_frames

    next_id = 10_000_000
    train_samples, next_id = frames_to_hand_samples(train_my_train_frames, next_id)
    val_samples, _ = frames_to_hand_samples(train_my_val_frames, next_id)

    split_mode = "train_my_only"

    if not train_samples:
        raise RuntimeError("Train samples are empty after excluding overlap with val_my.")
    if not val_samples:
        raise RuntimeError("Validation samples are empty. Check val_my annotations.")

    train_ds = HandKeypointDataset(
        samples=train_samples,
        image_size=IMAGE_SIZE,
        normalize_keypoints=NORMALIZE_KEYPOINTS,
        crop_margin=CROP_MARGIN,
    )
    val_ds = HandKeypointDataset(
        samples=val_samples,
        image_size=IMAGE_SIZE,
        normalize_keypoints=NORMALIZE_KEYPOINTS,
        crop_margin=CROP_MARGIN,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleHandKeypointCNN(num_keypoints=21).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCE_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        min_lr=MIN_LR,
    )

    best_val_loss = float("inf")
    no_improve_epochs = 0
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_mme": [],
        "val_mme": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
        "lr": [],
    }

    print(f"device: {device}")
    print(f"dataset_root: {DATASET_ROOT}")
    print(f"train samples: {len(train_ds)} | val samples: {len(val_ds)}")
    print(f"split_mode: {split_mode}")
    print(
        "split train_my | "
        f"frames_total={len(train_my_frames)} "
        f"train_frames={len(train_my_train_frames)} "
        f"val_frames={len(train_my_val_frames)} "
        f"overlap_removed={len(train_my_frames) - len(train_my_train_frames)}"
    )
    print(
        f"config | image_size={IMAGE_SIZE} batch_size={BATCH_SIZE} "
        f"epochs={EPOCHS} lr={LEARNING_RATE} "
        f"crop_margin={CROP_MARGIN} "
        f"lr_reduce_factor={LR_REDUCE_FACTOR} lr_plateau_patience={LR_PLATEAU_PATIENCE} "
        f"min_lr={MIN_LR} early_stop_patience={EARLY_STOP_PATIENCE}"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_mme_sum = 0.0
        train_mae_sum = 0.0
        train_mse_sum = 0.0
        train_visible_points = 0
        train_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            gt_xy = batch["keypoints"].to(device)
            vis = batch["visibility"].to(device)

            optimizer.zero_grad(set_to_none=True)

            pred_xy = model(images)
            visible_points_in_batch = int((vis > 0).sum().item())
            if visible_points_in_batch > 0:
                visible_mask = vis > 0
                loss = criterion(pred_xy[visible_mask], gt_xy[visible_mask])
            else:
                loss = criterion(pred_xy, gt_xy)

            loss.backward()
            optimizer.step()

            batch_mme_sum, batch_mae_sum, batch_mse_sum, batch_visible_points = (
                batch_visible_metric_sums(pred_xy, gt_xy, vis)
            )

            train_loss_sum += loss.item()
            train_mme_sum += batch_mme_sum
            train_mae_sum += batch_mae_sum
            train_mse_sum += batch_mse_sum
            train_visible_points += batch_visible_points
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_mme = train_mme_sum / max(train_visible_points, 1)
        train_mae = train_mae_sum / max(train_visible_points * 2, 1)
        train_rmse = math.sqrt(train_mse_sum / max(train_visible_points * 2, 1))

        model.eval()
        val_loss_sum = 0.0
        val_mme_sum = 0.0
        val_mae_sum = 0.0
        val_mse_sum = 0.0
        val_visible_points = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                gt_xy = batch["keypoints"].to(device)
                vis = batch["visibility"].to(device)

                pred_xy = model(images)
                visible_points_in_batch = int((vis > 0).sum().item())
                if visible_points_in_batch > 0:
                    visible_mask = vis > 0
                    val_loss = criterion(pred_xy[visible_mask], gt_xy[visible_mask])
                else:
                    val_loss = criterion(pred_xy, gt_xy)

                batch_mme_sum, batch_mae_sum, batch_mse_sum, batch_visible_points = (
                    batch_visible_metric_sums(pred_xy, gt_xy, vis)
                )

                val_loss_sum += val_loss.item()
                val_mme_sum += batch_mme_sum
                val_mae_sum += batch_mae_sum
                val_mse_sum += batch_mse_sum
                val_visible_points += batch_visible_points
                val_batches += 1

        val_loss_avg = val_loss_sum / max(val_batches, 1)
        val_mme_avg = val_mme_sum / max(val_visible_points, 1)
        val_mae_avg = val_mae_sum / max(val_visible_points * 2, 1)
        val_rmse_avg = math.sqrt(val_mse_sum / max(val_visible_points * 2, 1))
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss_avg)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_reduced = current_lr < prev_lr

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss_avg))
        history["train_mme"].append(float(train_mme))
        history["val_mme"].append(float(val_mme_avg))
        history["train_mae"].append(float(train_mae))
        history["val_mae"].append(float(val_mae_avg))
        history["train_rmse"].append(float(train_rmse))
        history["val_rmse"].append(float(val_rmse_avg))
        history["lr"].append(float(current_lr))

        print(
            f"epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} train_mme={train_mme:.6f} "
            f"train_mae={train_mae:.6f} train_rmse={train_rmse:.6f} | "
            f"val_loss={val_loss_avg:.6f} val_mme={val_mme_avg:.6f} "
            f"val_mae={val_mae_avg:.6f} val_rmse={val_rmse_avg:.6f} | "
            f"lr={current_lr:.8f}"
        )
        if lr_reduced:
            print(f"lr reduced: {prev_lr:.8f} -> {current_lr:.8f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_avg,
                },
                SAVE_PATH,
            )
            print(f"saved new best checkpoint -> {SAVE_PATH}")
        else:
            no_improve_epochs += 1
            if EARLY_STOP_PATIENCE > 0 and no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(
                    f"early stopping at epoch {epoch}: "
                    f"val_loss did not improve for {no_improve_epochs} epoch(s) "
                    f"(patience={EARLY_STOP_PATIENCE})"
                )
                break

    save_metrics_and_plots(history, SAVE_PATH)
    print(f"done. best_val_loss={best_val_loss:.6f}")
