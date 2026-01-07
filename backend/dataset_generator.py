import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2


def apply_filters(img, f):
    out = cv2.convertScaleAbs(img, alpha=f["contrast"], beta=f["brightness"])
    if f["blur"] > 1:
        out = cv2.GaussianBlur(out, (f["blur"], f["blur"]), 0)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an augmented dataset from base images stored in assets/Combined.\n"
            "Expected per base image: assets/Combined/{name}.png + assets/Combined/json/{name}.json + "
            "assets/Combined/yolo_label/{name}.txt.\n"
            "Outputs to: output/{name}/ + output/{name}/json/ + output/{name}/yolo_label/"
        )
    )

    parser.add_argument(
        "--combined-dir",
        default="assets/Combined",
        help="Folder containing base images and their json/yolo_label subfolders",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root output folder (created if missing)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="How many augmented images to create per base image",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated base image stems to process (example: forest_1,forest_2). If omitted, processes all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument("--brightness-min", type=int, default=-60)
    parser.add_argument("--brightness-max", type=int, default=60)
    parser.add_argument("--contrast-min", type=float, default=0.8)
    parser.add_argument("--contrast-max", type=float, default=1.3)
    parser.add_argument("--blur-min", type=int, default=0)
    parser.add_argument("--blur-max", type=int, default=7)

    return parser.parse_args()


def _odd_blur(v: int) -> int:
    if v <= 0:
        return 0
    return v if v % 2 == 1 else v + 1


def _iter_base_images(combined_dir: Path, stems: set[str] | None):
    for p in sorted(combined_dir.iterdir()):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if stems is not None and p.stem not in stems:
            continue
        yield p


def main() -> int:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    combined_dir = Path(args.combined_dir)
    json_dir = combined_dir / "json"
    yolo_dir = combined_dir / "yolo_label"

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stems = None
    if args.names:
        stems = {s.strip() for s in args.names.split(",") if s.strip()}

    processed = 0

    for base_img_path in _iter_base_images(combined_dir, stems):
        stem = base_img_path.stem
        base_json_path = json_dir / f"{stem}.json"
        base_yolo_path = yolo_dir / f"{stem}.txt"

        if not base_json_path.exists():
            print(f"[WARN] Missing base JSON for {stem}: {base_json_path}")
            continue
        if not base_yolo_path.exists():
            print(f"[WARN] Missing base YOLO label for {stem}: {base_yolo_path}")
            continue

        base_img = cv2.imread(str(base_img_path))
        if base_img is None:
            print(f"[WARN] Failed to read base image: {base_img_path}")
            continue

        with open(base_json_path, "r", encoding="utf-8") as f:
            base_meta = json.load(f)

        base_yolo_text = base_yolo_path.read_text(encoding="utf-8")

        out_dir = out_root / stem
        out_json_dir = out_dir / "json"
        out_yolo_dir = out_dir / "yolo_label"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_json_dir.mkdir(parents=True, exist_ok=True)
        out_yolo_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.count):
            aug = {
                "brightness": random.randint(args.brightness_min, args.brightness_max),
                "contrast": random.uniform(args.contrast_min, args.contrast_max),
                "blur": _odd_blur(random.randint(args.blur_min, args.blur_max)),
            }

            final_img = apply_filters(base_img, aug)

            b = aug["brightness"]
            c = int(aug["contrast"] * 100)
            bl = aug["blur"]

            img_name = (
                f"brightness{b:+04d}_"
                f"contrast{c:03d}_"
                f"blur{bl:02d}_"
                f"img{i:04d}"
            )

            out_img_path = out_dir / f"{img_name}.png"
            out_json_path = out_json_dir / f"{img_name}.json"
            out_yolo_path = out_yolo_dir / f"{img_name}.txt"

            ok = cv2.imwrite(str(out_img_path), final_img)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_img_path}")

            out_yolo_path.write_text(base_yolo_text, encoding="utf-8")

            out_meta = dict(base_meta)
            out_meta["base_source"] = {
                "image": base_img_path.name,
                "json": base_json_path.name,
                "yolo_label": base_yolo_path.name,
                "stem": stem,
            }
            out_meta["augmentation"] = aug
            out_meta["output"] = {
                "image": out_img_path.name,
                "json": out_json_path.name,
                "yolo_label": out_yolo_path.name,
            }
            out_meta["generated_at"] = datetime.now(timezone.utc).isoformat()

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(out_meta, f, indent=2)

        print(f"✅ Generated {args.count} images for base: {stem}")
        processed += 1

    if processed == 0:
        print("[WARN] No base images processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
