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
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=10,
        help="Epochs for images/train -> images/val pretraining. Use 0 to skip.",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=20,
        help="Epochs for images/train_my -> images/val_my fine-tuning. Use 0 to skip.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Backward-compatible alias for --finetune-epochs.",
    )
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
    if args.epochs is not None:
        args.finetune_epochs = args.epochs
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


def save_metrics_and_plots(history: Dict[str, List[Any]], save_path: Path) -> None:
    metrics_path = save_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"saved metrics -> {metrics_path}")

    if not history["global_epoch"]:
        print("skip plot: no epochs were run.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"skip plot: matplotlib is not available ({e})")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    phase_labels = {
        "pretrain": ("images/train", "images/val"),
        "finetune": ("train_my", "val_my"),
    }
    phases = []
    for phase in history["phase"]:
        if phase not in phases:
            phases.append(phase)

    def plot_metric_pair(ax: Any, train_key: str, val_key: str, ylabel: str) -> None:
        for phase in phases:
            idxs = [i for i, item in enumerate(history["phase"]) if item == phase]
            if not idxs:
                continue
            x = [history["global_epoch"][i] for i in idxs]
            train_label, val_label = phase_labels.get(phase, (f"{phase} train", f"{phase} val"))
            ax.plot(x, [history[train_key][i] for i in idxs], marker="o", label=f"{phase} {train_label}")
            ax.plot(x, [history[val_key][i] for i in idxs], marker="o", label=f"{phase} {val_label}")

        for i in range(1, len(history["phase"])):
            if history["phase"][i] != history["phase"][i - 1]:
                ax.axvline(history["global_epoch"][i] - 0.5, color="black", linestyle="--", alpha=0.35)

        ax.set_xlabel("Global epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_metric_pair(axes[0], "train_loss", "val_loss", "SmoothL1Loss")
    axes[0].set_title("Loss")

    plot_metric_pair(axes[1], "train_mme", "val_mme", "Mean Euclidean Error")
    axes[1].set_title("MME")

    for phase in phases:
        idxs = [i for i, item in enumerate(history["phase"]) if item == phase]
        if not idxs:
            continue
        x = [history["global_epoch"][i] for i in idxs]
        axes[2].plot(x, [history["train_mae"][i] for i in idxs], marker="o", label=f"{phase} train_mae")
        axes[2].plot(x, [history["val_mae"][i] for i in idxs], marker="o", label=f"{phase} val_mae")
        axes[2].plot(x, [history["train_rmse"][i] for i in idxs], marker="o", label=f"{phase} train_rmse")
        axes[2].plot(x, [history["val_rmse"][i] for i in idxs], marker="o", label=f"{phase} val_rmse")
    for i in range(1, len(history["phase"])):
        if history["phase"][i] != history["phase"][i - 1]:
            axes[2].axvline(history["global_epoch"][i] - 0.5, color="black", linestyle="--", alpha=0.35)
    axes[2].set_title("MAE / RMSE")
    axes[2].set_xlabel("Global epoch")
    axes[2].set_ylabel("Error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = save_path.with_suffix(".metrics.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"saved plot -> {plot_path}")


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training, but PyTorch does not see a CUDA GPU. "
            "Install a CUDA-enabled PyTorch build in this venv and check "
            "`python -c \"import torch; print(torch.cuda.is_available())\"`."
        )

    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def make_loader(
    samples: List[Dict[str, Any]],
    image_size: int,
    normalize_keypoints: bool,
    crop_margin: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    dataset = HandKeypointDataset(
        samples=samples,
        image_size=image_size,
        normalize_keypoints=normalize_keypoints,
        crop_margin=crop_margin,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    mme_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    visible_points = 0
    batches = 0
    non_blocking = device.type == "cuda"

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=non_blocking)
            gt_xy = batch["keypoints"].to(device, non_blocking=non_blocking)
            vis = batch["visibility"].to(device, non_blocking=non_blocking)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            pred_xy = model(images)
            visible_points_in_batch = int((vis > 0).sum().item())
            if visible_points_in_batch > 0:
                visible_mask = vis > 0
                loss = criterion(pred_xy[visible_mask], gt_xy[visible_mask])
            else:
                loss = criterion(pred_xy, gt_xy)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_mme_sum, batch_mae_sum, batch_mse_sum, batch_visible_points = (
                batch_visible_metric_sums(pred_xy, gt_xy, vis)
            )

            loss_sum += loss.item()
            mme_sum += batch_mme_sum
            mae_sum += batch_mae_sum
            mse_sum += batch_mse_sum
            visible_points += batch_visible_points
            batches += 1

    return {
        "loss": loss_sum / max(batches, 1),
        "mme": mme_sum / max(visible_points, 1),
        "mae": mae_sum / max(visible_points * 2, 1),
        "rmse": math.sqrt(mse_sum / max(visible_points * 2, 1)),
    }


def clone_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def init_history() -> Dict[str, List[Any]]:
    return {
        "phase": [],
        "phase_epoch": [],
        "global_epoch": [],
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


def append_history(
    history: Dict[str, List[Any]],
    phase: str,
    phase_epoch: int,
    global_epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
) -> None:
    history["phase"].append(phase)
    history["phase_epoch"].append(float(phase_epoch))
    history["global_epoch"].append(float(global_epoch))
    history["train_loss"].append(float(train_metrics["loss"]))
    history["val_loss"].append(float(val_metrics["loss"]))
    history["train_mme"].append(float(train_metrics["mme"]))
    history["val_mme"].append(float(val_metrics["mme"]))
    history["train_mae"].append(float(train_metrics["mae"]))
    history["val_mae"].append(float(val_metrics["mae"]))
    history["train_rmse"].append(float(train_metrics["rmse"]))
    history["val_rmse"].append(float(val_metrics["rmse"]))
    history["lr"].append(float(lr))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    phase: str,
    phase_epoch: int,
    global_epoch: int,
    val_loss: float,
    image_size: int,
    normalize_keypoints: bool,
    train_source: str,
    val_source: str,
) -> None:
    payload: Dict[str, Any] = {
        "phase": phase,
        "phase_epoch": phase_epoch,
        "global_epoch": global_epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "image_size": image_size,
        "normalize_keypoints": normalize_keypoints,
        "train_source": train_source,
        "val_source": val_source,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def run_training_phase(
    phase: str,
    train_source: str,
    val_source: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    lr_reduce_factor: float,
    lr_plateau_patience: int,
    min_lr: float,
    early_stop_patience: int,
    history: Dict[str, List[Any]],
    global_epoch_start: int,
    save_path: Path | None,
    image_size: int,
    normalize_keypoints: bool,
) -> tuple[int, Dict[str, torch.Tensor] | None, float, int]:
    if epochs <= 0:
        print(f"skip phase {phase}: epochs={epochs}")
        return global_epoch_start, None, float("inf"), 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_reduce_factor,
        patience=lr_plateau_patience,
        min_lr=min_lr,
    )

    best_state: Dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    global_epoch = global_epoch_start

    print(
        f"phase={phase} | train={train_source} val={val_source} "
        f"epochs={epochs} train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    for phase_epoch in range(1, epochs + 1):
        global_epoch += 1
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        lr_reduced = current_lr < prev_lr

        append_history(
            history=history,
            phase=phase,
            phase_epoch=phase_epoch,
            global_epoch=global_epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=current_lr,
        )

        print(
            f"{phase} epoch {phase_epoch:02d}/{epochs} "
            f"(global {global_epoch:02d}) | "
            f"train_loss={train_metrics['loss']:.6f} train_mme={train_metrics['mme']:.6f} "
            f"train_mae={train_metrics['mae']:.6f} train_rmse={train_metrics['rmse']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} val_mme={val_metrics['mme']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f} | "
            f"lr={current_lr:.8f}"
        )
        if lr_reduced:
            print(f"{phase} lr reduced: {prev_lr:.8f} -> {current_lr:.8f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = phase_epoch
            best_state = clone_model_state(model)
            no_improve_epochs = 0
            if save_path is not None:
                save_checkpoint(
                    path=save_path,
                    model=model,
                    optimizer=optimizer,
                    phase=phase,
                    phase_epoch=phase_epoch,
                    global_epoch=global_epoch,
                    val_loss=best_val_loss,
                    image_size=image_size,
                    normalize_keypoints=normalize_keypoints,
                    train_source=train_source,
                    val_source=val_source,
                )
                print(f"saved new best {phase} checkpoint -> {save_path}")
        else:
            no_improve_epochs += 1
            if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                print(
                    f"early stopping phase {phase} at epoch {phase_epoch}: "
                    f"val_loss did not improve for {no_improve_epochs} epoch(s) "
                    f"(patience={early_stop_patience})"
                )
                break

    return global_epoch, best_state, best_val_loss, best_epoch


def main() -> None:
    args = parse_args()

    DATASET_ROOT = resolve_dataset_root(args.dataset_root)
    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    PRETRAIN_EPOCHS = args.pretrain_epochs
    FINETUNE_EPOCHS = args.finetune_epochs
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

    if PRETRAIN_EPOCHS < 0 or FINETUNE_EPOCHS < 0:
        raise ValueError("--pretrain-epochs and --finetune-epochs must be >= 0")
    if PRETRAIN_EPOCHS == 0 and FINETUNE_EPOCHS == 0:
        raise ValueError("Nothing to train: both epoch counts are 0.")
    if CROP_MARGIN < 0:
        raise ValueError("--crop-margin must be >= 0")

    device = require_cuda_device()

    pretrain_train_samples = load_coco_split_samples(DATASET_ROOT, "train")
    pretrain_val_samples = load_coco_split_samples(DATASET_ROOT, "val")
    if PRETRAIN_EPOCHS > 0:
        if not pretrain_train_samples:
            raise RuntimeError("Pretrain train samples are empty. Check images/train and coco_annotation/train.")
        if not pretrain_val_samples:
            raise RuntimeError("Pretrain val samples are empty. Check images/val and coco_annotation/val.")

    train_my_frames = load_train_my_frame_records(DATASET_ROOT)
    val_my_frames = load_val_my_frame_records(DATASET_ROOT)
    if FINETUNE_EPOCHS > 0:
        if not train_my_frames:
            raise RuntimeError(
                "train_my is empty. Expected annotated data in images/train_my and images/train_my.annotations.jsonl"
            )
        if not val_my_frames:
            raise RuntimeError(
                "val_my is empty. Expected annotated data in images/val_my and images/val_my.annotations.jsonl"
            )

    val_names = {fr["image_name"] for fr in val_my_frames}
    train_my_train_frames = [fr for fr in train_my_frames if fr["image_name"] not in val_names]
    train_my_val_frames = val_my_frames

    next_id = 10_000_000
    finetune_train_samples, next_id = frames_to_hand_samples(train_my_train_frames, next_id)
    finetune_val_samples, _ = frames_to_hand_samples(train_my_val_frames, next_id)
    if FINETUNE_EPOCHS > 0:
        if not finetune_train_samples:
            raise RuntimeError("Fine-tune train samples are empty after excluding overlap with val_my.")
        if not finetune_val_samples:
            raise RuntimeError("Fine-tune val samples are empty. Check val_my annotations.")

    print(f"device: {device} | gpu: {torch.cuda.get_device_name(0)}")
    print(f"dataset_root: {DATASET_ROOT}")
    print(
        f"config | image_size={IMAGE_SIZE} batch_size={BATCH_SIZE} "
        f"pretrain_epochs={PRETRAIN_EPOCHS} finetune_epochs={FINETUNE_EPOCHS} "
        f"lr={LEARNING_RATE} crop_margin={CROP_MARGIN} "
        f"lr_reduce_factor={LR_REDUCE_FACTOR} lr_plateau_patience={LR_PLATEAU_PATIENCE} "
        f"min_lr={MIN_LR} early_stop_patience={EARLY_STOP_PATIENCE}"
    )
    print(
        "split pretrain | "
        f"train_samples={len(pretrain_train_samples)} val_samples={len(pretrain_val_samples)}"
    )
    print(
        "split finetune | "
        f"frames_total={len(train_my_frames)} "
        f"train_frames={len(train_my_train_frames)} "
        f"val_frames={len(train_my_val_frames)} "
        f"overlap_removed={len(train_my_frames) - len(train_my_train_frames)} "
        f"train_samples={len(finetune_train_samples)} val_samples={len(finetune_val_samples)}"
    )

    model = SimpleHandKeypointCNN(num_keypoints=21).to(device)
    criterion = nn.SmoothL1Loss()
    history = init_history()
    global_epoch = 0

    if PRETRAIN_EPOCHS > 0:
        pretrain_train_loader = make_loader(
            samples=pretrain_train_samples,
            image_size=IMAGE_SIZE,
            normalize_keypoints=NORMALIZE_KEYPOINTS,
            crop_margin=CROP_MARGIN,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=device,
            shuffle=True,
        )
        pretrain_val_loader = make_loader(
            samples=pretrain_val_samples,
            image_size=IMAGE_SIZE,
            normalize_keypoints=NORMALIZE_KEYPOINTS,
            crop_margin=CROP_MARGIN,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=device,
            shuffle=False,
        )
        global_epoch, pretrain_best_state, pretrain_best_loss, pretrain_best_epoch = run_training_phase(
            phase="pretrain",
            train_source="images/train",
            val_source="images/val",
            model=model,
            train_loader=pretrain_train_loader,
            val_loader=pretrain_val_loader,
            criterion=criterion,
            device=device,
            epochs=PRETRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_reduce_factor=LR_REDUCE_FACTOR,
            lr_plateau_patience=LR_PLATEAU_PATIENCE,
            min_lr=MIN_LR,
            early_stop_patience=EARLY_STOP_PATIENCE,
            history=history,
            global_epoch_start=global_epoch,
            save_path=SAVE_PATH if FINETUNE_EPOCHS == 0 else None,
            image_size=IMAGE_SIZE,
            normalize_keypoints=NORMALIZE_KEYPOINTS,
        )
        if pretrain_best_state is not None:
            model.load_state_dict(pretrain_best_state)
            print(
                f"loaded best pretrain state for fine-tune | "
                f"epoch={pretrain_best_epoch} val_loss={pretrain_best_loss:.6f}"
            )

    if FINETUNE_EPOCHS > 0:
        finetune_train_loader = make_loader(
            samples=finetune_train_samples,
            image_size=IMAGE_SIZE,
            normalize_keypoints=NORMALIZE_KEYPOINTS,
            crop_margin=CROP_MARGIN,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=device,
            shuffle=True,
        )
        finetune_val_loader = make_loader(
            samples=finetune_val_samples,
            image_size=IMAGE_SIZE,
            normalize_keypoints=NORMALIZE_KEYPOINTS,
            crop_margin=CROP_MARGIN,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=device,
            shuffle=False,
        )
        global_epoch, finetune_best_state, finetune_best_loss, finetune_best_epoch = run_training_phase(
            phase="finetune",
            train_source="images/train_my",
            val_source="images/val_my",
            model=model,
            train_loader=finetune_train_loader,
            val_loader=finetune_val_loader,
            criterion=criterion,
            device=device,
            epochs=FINETUNE_EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_reduce_factor=LR_REDUCE_FACTOR,
            lr_plateau_patience=LR_PLATEAU_PATIENCE,
            min_lr=MIN_LR,
            early_stop_patience=EARLY_STOP_PATIENCE,
            history=history,
            global_epoch_start=global_epoch,
            save_path=SAVE_PATH,
            image_size=IMAGE_SIZE,
            normalize_keypoints=NORMALIZE_KEYPOINTS,
        )
        if finetune_best_state is not None:
            model.load_state_dict(finetune_best_state)
            print(
                f"final checkpoint is best fine-tune state | "
                f"epoch={finetune_best_epoch} val_my_loss={finetune_best_loss:.6f}"
            )

    save_metrics_and_plots(history, SAVE_PATH)
    print(f"done. best checkpoint -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
