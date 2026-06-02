#!/usr/bin/env python3
"""
Evaluate DiffVS HEMIT with the **exact** Pix2pix_DualBranch post_process.py path.

1. Export each manifest pair as <stem>_real_B.tif / <stem>_fake_B.tif (same layout as test.py).
2. Run post_process.compute_metrics() via subprocess (identical code to dual-branch).

Set PIX2PIX_ROOT to your Pix2pix_DualBranch clone. Use --inline-only only if post_process is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from skimage.transform import resize

CSV_HEADER = [
    "file_name",
    "dapi_ssim",
    "cd3_ssim",
    "panck_ssim",
    "average_ssim",
    "dapi_pearson",
    "cd3_pearson",
    "panck_pearson",
    "average_pearson",
    "dapi_psnr",
    "cd3_psnr",
    "panck_psnr",
    "average_psnr",
]


def find_pix2pix_root(explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if (p / "post_process.py").is_file() else None
    env = os.environ.get("PIX2PIX_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "post_process.py").is_file():
            return p
    here = Path(__file__).resolve().parents[1]
    for candidate in (
        here.parent / "Pix2pix_DualBranch",
        here.parent / "pix2pix_dualbranch",
        Path.home() / "Pix2pix_DualBranch",
    ):
        if (candidate / "post_process.py").is_file():
            return candidate.resolve()
    return None


def load_rgb_array(path: Path) -> np.ndarray:
    arr = imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr.astype(np.float64)
    arr = arr.astype(np.float64)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0)


def resize_prediction_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == gt.shape[:2]:
        return pred
    out = resize(
        pred,
        gt.shape[:2],
        order=3,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(out, 0.0, 255.0)


def load_manifest(inference_dir: Path) -> list[dict]:
    manifest_path = inference_dir / "inference_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}. Run infer_hemit_diffusion_ft.sh first.")
    with open(manifest_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    return rows


def export_pix2pix_tiffs(rows: list[dict], export_dir: Path, do_resize: bool) -> int:
    """Write *_real_B.tif / *_fake_B.tif like test.py + post_process.py expect."""
    export_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in rows:
        target = Path(row["target_path"])
        pred = Path(row["prediction_path"])
        if not target.is_file() or not pred.is_file():
            print(f"[warn] skip missing pair: {target} / {pred}")
            continue
        stem = target.stem
        real = load_rgb_array(target)
        fake = load_rgb_array(pred)
        if do_resize:
            fake = resize_prediction_to_gt(fake, real)
        real_u8 = np.clip(real, 0, 255).astype(np.uint8)
        fake_u8 = np.clip(fake, 0, 255).astype(np.uint8)
        Image.fromarray(real_u8).save(export_dir / f"{stem}_real_B.tif")
        Image.fromarray(fake_u8).save(export_dir / f"{stem}_fake_B.tif")
        n += 1
    return n


def run_post_process(pix2pix_root: Path, export_dir: Path) -> Path:
    script = pix2pix_root / "post_process.py"
    print(f"Running dual-branch metrics: {script} --srcdir {export_dir}")
    subprocess.run(
        [sys.executable, str(script), "--srcdir", str(export_dir)],
        check=True,
        cwd=str(pix2pix_root),
    )
    score = export_dir / "score.csv"
    if not score.is_file():
        raise RuntimeError(f"post_process.py did not create {score}")
    return score


def print_score_summary(csv_path: Path) -> None:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    means = {}
    for key in rows[0]:
        if key == "file_name":
            continue
        vals = [float(r[key]) for r in rows]
        means[key] = sum(vals) / len(vals)
    print(f"Tiles: {len(rows)}")
    print(
        "Means — "
        f"SSIM dapi={means['dapi_ssim']:.4f} cd3={means['cd3_ssim']:.4f} "
        f"panck={means['panck_ssim']:.4f} avg={means['average_ssim']:.4f} | "
        f"Pearson avg={means['average_pearson']:.4f} | "
        f"PSNR avg={means['average_psnr']:.2f} dB"
    )


def run_inline_metrics(rows: list[dict], csv_path: Path, do_resize: bool) -> None:
    """Fallback: duplicated formulas (use post_process when possible)."""
    scored = 0
    tiny = 1e-15
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        for row in rows:
            target = Path(row["target_path"])
            pred = Path(row["prediction_path"])
            if not target.is_file() or not pred.is_file():
                continue
            real = load_rgb_array(target)
            fake = load_rgb_array(pred)
            if do_resize:
                fake = resize_prediction_to_gt(fake, real)
            ssim_scores, pearson_scores, psnr_scores = [], [], []
            for i in range(3):
                rc = real[:, :, i].astype(float)
                fc = fake[:, :, i].astype(float)
                rc[0, 0] += tiny
                fc[0, 0] += tiny
                ssim_scores.append(ssim(rc, fc, data_range=255))
                pearson_scores.append(float(np.corrcoef(rc.flatten(), fc.flatten())[0, 1]))
                psnr_scores.append(float(peak_signal_noise_ratio(rc, fc, data_range=255)))
            stem = target.stem
            writer.writerow(
                [
                    stem,
                    *ssim_scores,
                    float(np.mean(ssim_scores)),
                    *pearson_scores,
                    float(np.mean(pearson_scores)),
                    *psnr_scores,
                    float(np.mean(psnr_scores)),
                ]
            )
            scored += 1
    if scored == 0:
        raise RuntimeError("No pairs scored.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffVS HEMIT eval via Pix2pix post_process.py (directly comparable to dual-branch)."
    )
    parser.add_argument("--inference_dir", type=str, required=True)
    parser.add_argument(
        "--pix2pix-root",
        type=str,
        default="",
        help="Pix2pix_DualBranch root (or set PIX2PIX_ROOT). Default: auto-detect sibling repo.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="",
        help="TIFF export dir (default: <inference_dir>/pix2pix_metrics)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Copy of score.csv (default: <inference_dir>/score.csv)",
    )
    parser.add_argument(
        "--inline-only",
        action="store_true",
        help="Do not call post_process.py; use inlined copy of formulas",
    )
    parser.add_argument(
        "--no-resize-pred-to-gt",
        action="store_true",
        help="Keep DiffVS pred resolution (256²); not comparable to dual-branch @ 1024²",
    )
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir).expanduser().resolve()
    rows = load_manifest(inference_dir)
    do_resize = not args.no_resize_pred_to_gt
    export_dir = (
        Path(args.export_dir).expanduser().resolve()
        if args.export_dir
        else inference_dir / "pix2pix_metrics"
    )
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else inference_dir / "score.csv"

    n = export_pix2pix_tiffs(rows, export_dir, do_resize=do_resize)
    if n == 0:
        raise RuntimeError("No TIFF pairs exported.")
    print(f"Exported {n} pairs to {export_dir}")

    if args.inline_only:
        run_inline_metrics(rows, output_csv, do_resize=do_resize)
        print(f"Wrote {output_csv} (inline fallback)")
    else:
        pix2pix_root = find_pix2pix_root(args.pix2pix_root)
        if pix2pix_root is None:
            print(
                "WARN: Pix2pix_DualBranch not found; using inline metrics. "
                "Set PIX2PIX_ROOT=/path/to/Pix2pix_DualBranch for identical post_process.py.",
                file=sys.stderr,
            )
            run_inline_metrics(rows, output_csv, do_resize=do_resize)
        else:
            score_src = run_post_process(pix2pix_root, export_dir)
            shutil.copy2(score_src, output_csv)
            print(f"Copied {score_src} → {output_csv}")

    print_score_summary(output_csv)
    if do_resize:
        print("Compared at label resolution (1024²): preds upsampled to match dual-branch GT/fake TIFF size.")
    else:
        print("Compared at DiffVS native resolution (not directly comparable to dual-branch).")


if __name__ == "__main__":
    main()
