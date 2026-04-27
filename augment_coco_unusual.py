from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
class Sample:
    split: str
    image_id: int
    ann_id: int
    file_name: str
    width: int
    height: int
    category_id: int
    keypoints: list[float]
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unusual augmented COCO hand-keypoint dataset from train+val."
    )
    parser.add_argument(
        "--train-coco",
        type=str,
        default="coco_annotation/train/_annotations.coco.json",
    )
    parser.add_argument(
        "--val-coco",
        type=str,
        default="coco_annotation/val/_annotations.coco.json",
    )
    parser.add_argument("--images-root", type=str, default="images")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Fraction of samples sourced from train split.",
    )
    parser.add_argument(
        "--dst-images-dir",
        type=str,
        default="images/train_unusual_10k",
    )
    parser.add_argument(
        "--dst-coco",
        type=str,
        default="coco_annotation/train_unusual_10k/_annotations.coco.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-prob", type=float, default=0.8)
    parser.add_argument("--jpeg-quality-min", type=int, default=68)
    parser.add_argument("--jpeg-quality-max", type=int, default=95)
    parser.add_argument("--jpeg-compress-prob", type=float, default=0.45)
    args, _unknown = parser.parse_known_args()
    return args


def read_image_bgr(path: Path) -> np.ndarray | None:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def write_jpeg(path: Path, image_bgr: np.ndarray, quality: int) -> bool:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


def load_coco_samples(coco_path: Path, split: str, images_root: Path) -> tuple[list[Sample], dict[str, Any]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    anns = data.get("annotations", [])

    img_by_id = {int(x["id"]): x for x in images}
    samples: list[Sample] = []

    for ann in anns:
        k = ann.get("keypoints", [])
        if not isinstance(k, list) or len(k) != 63:
            continue
        image_id = int(ann["image_id"])
        info = img_by_id.get(image_id)
        if info is None:
            continue
        file_name = str(info["file_name"])
        image_path = images_root / split / file_name
        if not image_path.exists():
            continue
        samples.append(
            Sample(
                split=split,
                image_id=image_id,
                ann_id=int(ann["id"]),
                file_name=file_name,
                width=int(info["width"]),
                height=int(info["height"]),
                category_id=int(ann.get("category_id", 1)),
                keypoints=[float(x) for x in k],
                image_path=image_path,
            )
        )

    return samples, data


def kp_list_to_arrays(keypoints: list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(keypoints, dtype=np.float32).reshape(21, 3)
    xy = arr[:, :2].copy()
    v = arr[:, 2].copy()
    return xy, v


def apply_affine_to_points(points_xy: np.ndarray, m2x3: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)
    homo = np.hstack([points_xy.astype(np.float32), ones])
    return (homo @ m2x3.T).astype(np.float32)


def random_synthetic_background(h: int, w: int, rng: random.Random) -> np.ndarray:
    mode = rng.randint(0, 2)
    if mode == 0:
        base = np.full((h, w, 3), rng.uniform(25, 230), dtype=np.float32)
        noise = np.random.normal(0.0, rng.uniform(5.0, 30.0), size=base.shape).astype(np.float32)
        out = base + noise
    elif mode == 1:
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        c1 = np.array([rng.uniform(20, 235), rng.uniform(20, 235), rng.uniform(20, 235)], dtype=np.float32)
        c2 = np.array([rng.uniform(20, 235), rng.uniform(20, 235), rng.uniform(20, 235)], dtype=np.float32)
        mix = rng.uniform(0.2, 0.8)
        grad = mix * x + (1.0 - mix) * y
        out = c1 * (1.0 - grad) + c2 * grad
    else:
        out = np.random.uniform(0, 255, size=(h, w, 3)).astype(np.float32)
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=rng.uniform(8.0, 25.0))
    return np.clip(out, 0, 255).astype(np.uint8)


def build_hand_mask(points_xy: np.ndarray, vis: np.ndarray, h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = vis > 0
    if valid.sum() < 3:
        return mask.astype(np.float32)

    pts = points_xy.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = np.round(pts).astype(np.int32)

    hull = cv2.convexHull(pts_i[valid])
    if hull.shape[0] >= 3:
        cv2.fillConvexPoly(mask, hull, 255)

    thickness = max(6, int(0.02 * max(h, w)))
    radius = max(4, int(0.012 * max(h, w)))
    for a, b in HAND_CONNECTIONS:
        if not (valid[a] and valid[b]):
            continue
        p1 = (int(pts_i[a, 0]), int(pts_i[a, 1]))
        p2 = (int(pts_i[b, 0]), int(pts_i[b, 1]))
        cv2.line(mask, p1, p2, 255, thickness, cv2.LINE_AA)
    for idx in np.where(valid)[0]:
        p = pts_i[idx]
        cv2.circle(mask, (int(p[0]), int(p[1])), radius, 255, -1, cv2.LINE_AA)

    ksize = int(rng.choice([9, 11, 13, 15, 17]))
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(2.0, 4.0))
    return np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)


def augment_sample(
    src_img: np.ndarray,
    keypoints: list[float],
    bg_pool: list[Path],
    rng: random.Random,
    background_prob: float,
    jpeg_compress_prob: float,
    jpeg_quality_min: int,
    jpeg_quality_max: int,
) -> tuple[np.ndarray, list[float], list[float], bool]:
    h, w = src_img.shape[:2]
    xy, vis = kp_list_to_arrays(keypoints)

    angle = rng.uniform(-22.0, 22.0)
    scale = rng.uniform(0.82, 1.18)
    tx = rng.uniform(-0.1 * w, 0.1 * w)
    ty = rng.uniform(-0.1 * h, 0.1 * h)
    center = (w * 0.5, h * 0.5)
    m = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    m[0, 2] += tx
    m[1, 2] += ty

    border = int(rng.uniform(90, 170))
    out = cv2.warpAffine(
        src_img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border, border, border),
    )
    xy = apply_affine_to_points(xy, m)

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        xy[:, 0] = (w - 1) - xy[:, 0]

    in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    vis = vis * in_bounds.astype(np.float32)

    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)

    if vis.sum() < 8:
        return out, [], [], False

    if rng.random() < background_prob:
        alpha = build_hand_mask(xy, vis, h, w, rng)
        if bg_pool and rng.random() < 0.8:
            bg_path = random.choice(bg_pool)
            bg_img = read_image_bgr(bg_path)
            if bg_img is not None:
                bg = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                bg = random_synthetic_background(h, w, rng)
        else:
            bg = random_synthetic_background(h, w, rng)
        out = np.clip(
            out.astype(np.float32) * alpha[..., None] + bg.astype(np.float32) * (1.0 - alpha[..., None]),
            0,
            255,
        ).astype(np.uint8)

    img = out.astype(np.float32)
    alpha_bc = rng.uniform(0.8, 1.22)
    beta_bc = rng.uniform(-30.0, 30.0)
    img = img * alpha_bc + beta_bc

    gamma = rng.uniform(0.78, 1.28)
    img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

    if rng.random() < 0.32:
        k = 3 if rng.random() < 0.5 else 5
        img = cv2.GaussianBlur(img, (k, k), sigmaX=0.0)

    if rng.random() < 0.35:
        sigma = rng.uniform(3.0, 14.0)
        noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
        img = img + noise

    if rng.random() < 0.25:
        occ_w = int(rng.uniform(0.08, 0.2) * w)
        occ_h = int(rng.uniform(0.08, 0.2) * h)
        x0 = int(rng.uniform(0, max(1, w - occ_w)))
        y0 = int(rng.uniform(0, max(1, h - occ_h)))
        fill = float(rng.uniform(55, 210))
        img[y0 : y0 + occ_h, x0 : x0 + occ_w] = fill

    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < jpeg_compress_prob:
        q = int(rng.uniform(jpeg_quality_min, jpeg_quality_max))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                img = dec

    kp_out: list[float] = []
    for i in range(21):
        if vis[i] > 0:
            kp_out.extend([float(xy[i, 0]), float(xy[i, 1]), float(vis[i])])
        else:
            kp_out.extend([0.0, 0.0, 0.0])

    visible_idx = np.where(vis > 0)[0]
    vx = xy[visible_idx, 0]
    vy = xy[visible_idx, 1]
    x1 = float(vx.min())
    y1 = float(vy.min())
    x2 = float(vx.max())
    y2 = float(vy.max())
    bbox = [x1, y1, float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))]
    return img, kp_out, bbox, True


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if not (0.0 <= args.train_ratio <= 1.0):
        raise ValueError("--train-ratio must be in [0..1]")
    if not (0.0 <= args.background_prob <= 1.0):
        raise ValueError("--background-prob must be in [0..1]")
    if not (0.0 <= args.jpeg_compress_prob <= 1.0):
        raise ValueError("--jpeg-compress-prob must be in [0..1]")
    if not (1 <= args.jpeg_quality_min <= 100 and 1 <= args.jpeg_quality_max <= 100):
        raise ValueError("--jpeg-quality-min/max must be in [1..100]")
    if args.jpeg_quality_min > args.jpeg_quality_max:
        raise ValueError("--jpeg-quality-min must be <= --jpeg-quality-max")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    root = Path.cwd()
    train_coco = (root / args.train_coco).resolve()
    val_coco = (root / args.val_coco).resolve()
    images_root = (root / args.images_root).resolve()
    dst_images_dir = (root / args.dst_images_dir).resolve()
    dst_coco = (root / args.dst_coco).resolve()
    dst_coco.parent.mkdir(parents=True, exist_ok=True)
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    train_samples, train_data = load_coco_samples(train_coco, "train", images_root)
    val_samples, _val_data = load_coco_samples(val_coco, "val", images_root)
    if not train_samples or not val_samples:
        raise RuntimeError("Failed to load train/val samples from COCO.")

    print(f"train samples loaded: {len(train_samples)}")
    print(f"val samples loaded: {len(val_samples)}")

    all_source_paths = [s.image_path for s in train_samples] + [s.image_path for s in val_samples]
    bg_pool = all_source_paths

    n_train = int(round(args.num_samples * args.train_ratio))
    n_val = args.num_samples - n_train
    print(f"target unusual samples: {args.num_samples} (train-src={n_train}, val-src={n_val})")

    # clean old output files for deterministic ids
    for old in dst_images_dir.glob("unusual_*.jpg"):
        old.unlink(missing_ok=True)

    out_images: list[dict[str, Any]] = []
    out_annotations: list[dict[str, Any]] = []
    categories = train_data.get("categories", [{"id": 1, "name": "hand", "keypoints": [], "skeleton": []}])

    produced = 0
    ann_id = 1
    attempts = 0
    max_attempts = args.num_samples * 20

    while produced < args.num_samples and attempts < max_attempts:
        attempts += 1
        use_train = produced < n_train if (produced < n_train) else False
        if produced >= n_train:
            use_train = False
        if produced < n_train and produced + (n_val) >= args.num_samples:
            use_train = True

        # keep requested ratio by counting already produced from each bucket
        train_done = sum(1 for x in out_images if x.get("_src") == "train")
        val_done = sum(1 for x in out_images if x.get("_src") == "val")
        if train_done >= n_train:
            use_train = False
        elif val_done >= n_val:
            use_train = True

        src = random.choice(train_samples if use_train else val_samples)
        src_img = read_image_bgr(src.image_path)
        if src_img is None:
            continue
        if src_img.shape[0] != src.height or src_img.shape[1] != src.width:
            src_img = cv2.resize(src_img, (src.width, src.height), interpolation=cv2.INTER_LINEAR)

        aug_img, kp_out, bbox, ok = augment_sample(
            src_img=src_img,
            keypoints=src.keypoints,
            bg_pool=bg_pool,
            rng=rng,
            background_prob=args.background_prob,
            jpeg_compress_prob=args.jpeg_compress_prob,
            jpeg_quality_min=args.jpeg_quality_min,
            jpeg_quality_max=args.jpeg_quality_max,
        )
        if not ok:
            continue

        file_name = f"unusual_{produced:05d}.jpg"
        out_path = dst_images_dir / file_name
        qsave = rng.randint(args.jpeg_quality_min, args.jpeg_quality_max)
        if not write_jpeg(out_path, aug_img, qsave):
            continue

        img_id = produced + 1
        out_images.append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": int(src.width),
                "height": int(src.height),
                "_src": src.split,
            }
        )
        area = float(max(1.0, bbox[2] * bbox[3]))
        out_annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(src.category_id),
                "bbox": [float(x) for x in bbox],
                "area": area,
                "segmentation": [],
                "iscrowd": 0,
                "keypoints": [float(x) for x in kp_out],
                "num_keypoints": int(sum(1 for i in range(2, 63, 3) if kp_out[i] > 0)),
            }
        )
        ann_id += 1
        produced += 1

        if produced % 500 == 0:
            print(f"generated {produced}/{args.num_samples} ...")

    if produced < args.num_samples:
        print(f"warning: generated {produced}/{args.num_samples} after {attempts} attempts")

    # remove helper key before writing
    for x in out_images:
        x.pop("_src", None)

    coco_out = {
        "info": {
            "description": "Unusual augmented hand dataset from train+val",
            "version": "1.0",
        },
        "licenses": train_data.get("licenses", []),
        "categories": categories,
        "images": out_images,
        "annotations": out_annotations,
    }
    dst_coco.write_text(json.dumps(coco_out, ensure_ascii=False), encoding="utf-8")
    print(f"done. images: {len(out_images)} annotations: {len(out_annotations)}")
    print(f"saved images dir: {dst_images_dir}")
    print(f"saved coco: {dst_coco}")


if __name__ == "__main__":
    main()

