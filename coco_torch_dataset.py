from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HandKeypointCocoDataset(Dataset):
    """
    COCO keypoint dataset for one-hand-per-image annotations.

    Output sample:
    {
        "image": Tensor[C,H,W], float32 in [0,1],
        "keypoints": Tensor[K,2], float32 (x,y),
        "visibility": Tensor[K], float32 (v),
        "image_id": int,
        "file_name": str
    }
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        normalize_keypoints: bool = False,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.normalize_keypoints = normalize_keypoints
        self.image_transform = image_transform

        self.images_dir = self.dataset_root / "images" / split
        self.annotations_path = (
            self.dataset_root / "coco_annotation" / split / "_annotations.coco.json"
        )

        if not self.annotations_path.exists():
            raise FileNotFoundError(f"COCO json not found: {self.annotations_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        with self.annotations_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        ann_by_image_id = {ann["image_id"]: ann for ann in coco["annotations"]}

        self.samples: List[Dict[str, Any]] = []
        for img in coco["images"]:
            img_id = img["id"]
            ann = ann_by_image_id.get(img_id)
            if ann is None:
                continue

            keypoints = ann.get("keypoints", [])
            if len(keypoints) == 0 or len(keypoints) % 3 != 0:
                continue

            file_name = img["file_name"]
            image_path = self.images_dir / file_name
            if not image_path.exists():
                continue

            self.samples.append(
                {
                    "image_id": img_id,
                    "file_name": file_name,
                    "image_path": image_path,
                    "width": img["width"],
                    "height": img["height"],
                    "keypoints": keypoints,
                }
            )

        if not self.samples:
            raise RuntimeError(
                "No valid samples were built. Check paths and annotation format."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        if self.image_transform is not None:
            image_tensor = self.image_transform(image_tensor)

        kp = torch.tensor(sample["keypoints"], dtype=torch.float32).view(-1, 3)
        keypoints_xy = kp[:, :2]
        visibility = kp[:, 2]

        if self.normalize_keypoints:
            w = float(sample["width"])
            h = float(sample["height"])
            keypoints_xy = keypoints_xy / torch.tensor([w, h], dtype=torch.float32)

        return {
            "image": image_tensor,
            "keypoints": keypoints_xy,
            "visibility": visibility,
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
        }


def _demo() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Path to dataset root (contains images/, coco_annotation/, labels/). "
            "If omitted, uses folder where this script is located."
        ),
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--normalize-keypoints", action="store_true")
    args = parser.parse_args()

    dataset_root = (
        Path(args.dataset_root)
        if args.dataset_root is not None
        else Path(__file__).resolve().parent
    )

    ds = HandKeypointCocoDataset(
        dataset_root=dataset_root,
        split=args.split,
        normalize_keypoints=args.normalize_keypoints,
    )

    first = ds[0]
    print(f"samples: {len(ds)}")
    print(f"file_name: {first['file_name']}")
    print(f"image shape: {tuple(first['image'].shape)}")
    print(f"keypoints shape: {tuple(first['keypoints'].shape)}")
    print(f"visibility shape: {tuple(first['visibility'].shape)}")
    print(f"first keypoint: {first['keypoints'][0].tolist()}, v={first['visibility'][0].item()}")


if __name__ == "__main__":
    _demo()
