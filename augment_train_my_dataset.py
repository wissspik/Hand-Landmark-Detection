from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from datetime import datetime, timezone
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment train_my dataset with keypoint-aware transforms and write "
            "results to images/train_my_aug."
        )
    )
    parser.add_argument("--src-images-dir", type=str, default="images/train_my")
    parser.add_argument(
        "--src-annotations", type=str, default="images/train_my.annotations.jsonl"
    )
    parser.add_argument("--dst-images-dir", type=str, default="images/train_my_aug")
    parser.add_argument(
        "--dst-annotations", type=str, default="images/train_my_aug.annotations.jsonl"
    )
    parser.add_argument("--variants-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-originals",
        dest="include_originals",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-include-originals",
        dest="include_originals",
        action="store_false",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N unique source images (for quick tests).",
    )
    parser.add_argument(
        "--background-prob",
        type=float,
        default=0.6,
        help="Probability of replacing background in each augmented sample.",
    )
    parser.add_argument(
        "--background-images-dir",
        type=str,
        default=None,
        help="Optional folder with background images for compositing.",
    )
    parser.add_argument(
        "--jpeg-quality-min",
        type=int,
        default=70,
        help="Min JPEG quality for saved augmented images.",
    )
    parser.add_argument(
        "--jpeg-quality-max",
        type=int,
        default=95,
        help="Max JPEG quality for saved augmented images.",
    )
    parser.add_argument(
        "--jpeg-compress-prob",
        type=float,
        default=0.45,
        help="Extra in-pipeline JPEG degradation probability.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def write_jpeg(frame_bgr: np.ndarray, path: Path, quality: int = 95) -> bool:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def read_image_bgr(path: Path) -> np.ndarray | None:
    # cv2.imread may fail on Windows unicode paths; fromfile+imdecode is safer.
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def load_unique_records(annotations_path: Path) -> tuple[list[dict[str, Any]], int]:
    raw_records: list[dict[str, Any]] = []
    with annotations_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip bad json line {i}: {e}")
                continue
            if "image" not in rec:
                continue
            raw_records.append(rec)

    seen: set[str] = set()
    unique_reversed: list[dict[str, Any]] = []
    for rec in reversed(raw_records):
        name = str(rec["image"])
        if name in seen:
            continue
        seen.add(name)
        unique_reversed.append(rec)

    unique_records = list(reversed(unique_reversed))
    duplicates = len(raw_records) - len(unique_records)
    return unique_records, duplicates


def parse_hand_points(hand: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, list[float]]:
    landmarks_px = hand.get("landmarks_px", [])
    landmarks_norm = hand.get("landmarks_norm", [])

    points: list[list[float]] = []
    z_vals: list[float] = []

    if isinstance(landmarks_px, list) and len(landmarks_px) == 21:
        for i, p in enumerate(landmarks_px):
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            points.append([x, y])
            if isinstance(landmarks_norm, list) and len(landmarks_norm) == 21:
                z_vals.append(float(landmarks_norm[i].get("z", 0.0)))
            else:
                z_vals.append(0.0)
    elif isinstance(landmarks_norm, list) and len(landmarks_norm) == 21:
        for p in landmarks_norm:
            x = float(p.get("x", 0.0)) * width
            y = float(p.get("y", 0.0)) * height
            z = float(p.get("z", 0.0))
            points.append([x, y])
            z_vals.append(z)

    if len(points) != 21:
        return np.zeros((0, 2), dtype=np.float32), []
    return np.asarray(points, dtype=np.float32), z_vals


def apply_affine_to_points(points_xy: np.ndarray, matrix_2x3: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.copy()
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)
    homo = np.hstack([points_xy.astype(np.float32), ones])
    out = homo @ matrix_2x3.T
    return out.astype(np.float32)


def swap_handedness(label: str | None) -> str | None:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def collect_background_pool(background_images_dir: Path | None) -> list[Path]:
    if background_images_dir is None:
        return []
    if not background_images_dir.exists():
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: list[Path] = []
    for ext in exts:
        files.extend(background_images_dir.rglob(ext))
    return files


def random_synthetic_background(h: int, w: int, rng: random.Random) -> np.ndarray:
    mode = rng.randint(0, 2)
    if mode == 0:
        base = np.full((h, w, 3), rng.uniform(30, 220), dtype=np.float32)
        noise = np.random.normal(0.0, rng.uniform(5.0, 25.0), size=base.shape).astype(np.float32)
        out = base + noise
    elif mode == 1:
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        c1 = np.array([rng.uniform(20, 235), rng.uniform(20, 235), rng.uniform(20, 235)], dtype=np.float32)
        c2 = np.array([rng.uniform(20, 235), rng.uniform(20, 235), rng.uniform(20, 235)], dtype=np.float32)
        mix = rng.uniform(0.3, 0.7)
        grad = mix * x + (1.0 - mix) * y
        out = c1 * (1.0 - grad) + c2 * grad
    else:
        out = np.random.uniform(0, 255, size=(h, w, 3)).astype(np.float32)
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=rng.uniform(8.0, 25.0))
    return np.clip(out, 0, 255).astype(np.uint8)


def sample_background_image(
    h: int,
    w: int,
    rng: random.Random,
    background_pool: list[Path],
) -> np.ndarray:
    if background_pool and rng.random() < 0.8:
        bg_path = random.choice(background_pool)
        bg = read_image_bgr(bg_path)
        if bg is not None:
            return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return random_synthetic_background(h, w, rng)


def build_hand_mask_from_landmarks(
    out_hands: list[dict[str, Any]],
    h: int,
    w: int,
    rng: random.Random,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    thickness = max(6, int(0.02 * max(h, w)))
    radius = max(4, int(0.012 * max(h, w)))

    for hand in out_hands:
        landmarks = hand.get("landmarks_px", [])
        if not isinstance(landmarks, list) or len(landmarks) != 21:
            continue

        pts = np.array(
            [[float(p.get("x", 0.0)), float(p.get("y", 0.0))] for p in landmarks],
            dtype=np.float32,
        )
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = np.round(pts).astype(np.int32)

        hull = cv2.convexHull(pts_i)
        if hull.shape[0] >= 3:
            cv2.fillConvexPoly(mask, hull, 255)

        for a, b in HAND_CONNECTIONS:
            p1 = (int(pts_i[a, 0]), int(pts_i[a, 1]))
            p2 = (int(pts_i[b, 0]), int(pts_i[b, 1]))
            cv2.line(mask, p1, p2, 255, thickness, cv2.LINE_AA)
        for p in pts_i:
            cv2.circle(mask, (int(p[0]), int(p[1])), radius, 255, -1, cv2.LINE_AA)

    ksize = int(rng.choice([9, 11, 13, 15, 17]))
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(2.0, 4.0))
    alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
    return alpha


def build_augmented_sample(
    image_bgr: np.ndarray,
    hands: list[dict[str, Any]],
    rng: random.Random,
    background_pool: list[Path],
    background_prob: float,
    jpeg_compress_prob: float,
    jpeg_quality_min: int,
    jpeg_quality_max: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h, w = image_bgr.shape[:2]

    out_img = image_bgr.copy()
    out_hands: list[dict[str, Any]] = []

    angle = rng.uniform(-20.0, 20.0)
    scale = rng.uniform(0.85, 1.15)
    tx = rng.uniform(-0.08 * w, 0.08 * w)
    ty = rng.uniform(-0.08 * h, 0.08 * h)
    center = (w * 0.5, h * 0.5)
    m = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    m[0, 2] += tx
    m[1, 2] += ty

    border = int(rng.uniform(95, 160))
    out_img = cv2.warpAffine(
        out_img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border, border, border),
    )

    do_flip = rng.random() < 0.5

    for hand in hands:
        points = hand["points_xy"]
        z_vals = hand["z_vals"]
        transformed = apply_affine_to_points(points, m)

        label = hand["label"]
        if do_flip:
            transformed[:, 0] = (w - 1) - transformed[:, 0]
            label = swap_handedness(label)

        transformed[:, 0] = np.clip(transformed[:, 0], 0, w - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, h - 1)

        landmarks_px: list[dict[str, float]] = []
        landmarks_norm: list[dict[str, float]] = []
        for i in range(21):
            x = float(transformed[i, 0])
            y = float(transformed[i, 1])
            z = float(z_vals[i]) if i < len(z_vals) else 0.0
            landmarks_px.append({"x": x, "y": y})
            landmarks_norm.append({"x": x / float(w), "y": y / float(h), "z": z})

        out_hands.append(
            {
                "hand_index": int(hand["hand_index"]),
                "label": label,
                "score": hand["score"],
                "landmarks_norm": landmarks_norm,
                "landmarks_px": landmarks_px,
            }
        )

    if do_flip:
        out_img = cv2.flip(out_img, 1)

    if out_hands and background_prob > 0 and rng.random() < background_prob:
        alpha = build_hand_mask_from_landmarks(out_hands, h, w, rng)
        bg = sample_background_image(h, w, rng, background_pool).astype(np.float32)
        fg = out_img.astype(np.float32)
        out_img = np.clip(
            fg * alpha[..., None] + bg * (1.0 - alpha[..., None]),
            0,
            255,
        ).astype(np.uint8)

    img = out_img.astype(np.float32)
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(-25.0, 25.0)
    img = img * alpha + beta

    gamma = rng.uniform(0.8, 1.25)
    img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

    if rng.random() < 0.3:
        k = 3 if rng.random() < 0.5 else 5
        img = cv2.GaussianBlur(img, (k, k), sigmaX=0.0)

    if rng.random() < 0.35:
        sigma = rng.uniform(3.0, 12.0)
        noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
        img = img + noise

    if rng.random() < 0.25:
        occ_w = int(rng.uniform(0.08, 0.2) * w)
        occ_h = int(rng.uniform(0.08, 0.2) * h)
        x0 = int(rng.uniform(0, max(1, w - occ_w)))
        y0 = int(rng.uniform(0, max(1, h - occ_h)))
        fill = float(rng.uniform(60, 200))
        img[y0 : y0 + occ_h, x0 : x0 + occ_w] = fill

    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < jpeg_compress_prob:
        qmin = max(20, min(100, jpeg_quality_min))
        qmax = max(qmin, min(100, jpeg_quality_max))
        quality = int(rng.uniform(qmin, qmax))
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                img = decoded

    return img, out_hands


def make_aug_name(image_name: str, aug_idx: int) -> str:
    p = Path(image_name)
    ext = p.suffix if p.suffix else ".jpg"
    stem = p.stem
    return f"{stem}_aug{aug_idx:02d}{ext}"


def main() -> None:
    args = parse_args()
    if args.variants_per_image < 1:
        raise ValueError("--variants-per-image must be >= 1")
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

    src_images_dir = Path(args.src_images_dir).expanduser().resolve()
    src_annotations = Path(args.src_annotations).expanduser().resolve()
    dst_images_dir = Path(args.dst_images_dir).expanduser().resolve()
    dst_annotations = Path(args.dst_annotations).expanduser().resolve()
    background_images_dir = (
        Path(args.background_images_dir).expanduser().resolve()
        if args.background_images_dir
        else None
    )

    if not src_images_dir.exists():
        raise FileNotFoundError(f"src images dir not found: {src_images_dir}")
    if not src_annotations.exists():
        raise FileNotFoundError(f"src annotations not found: {src_annotations}")

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_annotations.parent.mkdir(parents=True, exist_ok=True)

    records, duplicate_count = load_unique_records(src_annotations)
    if args.limit is not None:
        records = records[: args.limit]

    background_pool = collect_background_pool(background_images_dir)

    print(f"src images dir: {src_images_dir}")
    print(f"dst images dir: {dst_images_dir}")
    print(f"source records (unique by image): {len(records)}")
    print(f"duplicates removed from source annotations: {duplicate_count}")
    if background_images_dir:
        print(f"background images dir: {background_images_dir}")
        print(f"background pool size: {len(background_pool)}")
    else:
        print("background mode: synthetic")

    saved_images = 0
    skipped = 0
    out_records: list[dict[str, Any]] = []

    for rec in records:
        image_name = str(rec.get("image", ""))
        src_img_path = src_images_dir / image_name
        if not src_img_path.exists():
            skipped += 1
            continue

        img = read_image_bgr(src_img_path)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        hand_records = rec.get("hands", [])
        parsed_hands: list[dict[str, Any]] = []
        for hand in hand_records:
            points_xy, z_vals = parse_hand_points(hand, w, h)
            if points_xy.shape[0] != 21:
                continue
            parsed_hands.append(
                {
                    "hand_index": int(hand.get("hand_index", len(parsed_hands))),
                    "label": hand.get("label"),
                    "score": hand.get("score"),
                    "points_xy": points_xy,
                    "z_vals": z_vals,
                }
            )

        if args.include_originals:
            dst_orig_path = dst_images_dir / image_name
            dst_orig_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img_path, dst_orig_path)
            out_records.append(
                {
                    "image": image_name,
                    "width": int(w),
                    "height": int(h),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "num_hands": len(parsed_hands),
                    "hands": [
                        {
                            "hand_index": int(hd["hand_index"]),
                            "label": hd["label"],
                            "score": hd["score"],
                            "landmarks_norm": [
                                {"x": float(p[0]) / float(w), "y": float(p[1]) / float(h), "z": float(z)}
                                for p, z in zip(hd["points_xy"], hd["z_vals"])
                            ],
                            "landmarks_px": [
                                {"x": float(p[0]), "y": float(p[1])}
                                for p in hd["points_xy"]
                            ],
                        }
                        for hd in parsed_hands
                    ],
                }
            )
            saved_images += 1

        for aug_idx in range(1, args.variants_per_image + 1):
            aug_img, aug_hands = build_augmented_sample(
                img,
                parsed_hands,
                rng,
                background_pool=background_pool,
                background_prob=args.background_prob,
                jpeg_compress_prob=args.jpeg_compress_prob,
                jpeg_quality_min=args.jpeg_quality_min,
                jpeg_quality_max=args.jpeg_quality_max,
            )
            aug_name = make_aug_name(image_name, aug_idx)
            dst_aug_path = dst_images_dir / aug_name

            out_quality = rng.randint(args.jpeg_quality_min, args.jpeg_quality_max)
            ok = write_jpeg(aug_img, dst_aug_path, quality=out_quality)
            if not ok:
                skipped += 1
                continue

            out_records.append(
                {
                    "image": aug_name,
                    "width": int(w),
                    "height": int(h),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "num_hands": len(aug_hands),
                    "hands": aug_hands,
                }
            )
            saved_images += 1

    with dst_annotations.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"done. saved images: {saved_images}")
    print(f"done. annotation records: {len(out_records)}")
    print(f"skipped samples: {skipped}")
    print(f"dst annotations: {dst_annotations}")


if __name__ == "__main__":
    main()
