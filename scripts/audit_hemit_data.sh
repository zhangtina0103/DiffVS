#!/usr/bin/env bash
# Save one train/test pair as PNGs so you can confirm input=H&E and label=mIHC (not swapped).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/data}"
OUT="${OUT:-${REPO_DIR}/outputs/hemit_data_audit}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
mkdir -p "${OUT}"

python - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "${REPO_DIR}/src")
from torchvision.transforms.functional import to_pil_image
from diffvs.datasets import build_dataset

root = Path("${DATASET_ROOT}")
out = Path("${OUT}")
for split in ("train", "test"):
    try:
        ds, _ = build_dataset("hemit", str(root), split, 256, ["HEMIT"], max_rows=1)
    except Exception as e:
        print(f"[{split}] SKIP: {e}")
        continue
    b = ds[0]
    to_pil_image(b["source"].cpu()).save(out / f"{split}_00_input_HE.png")
    to_pil_image(b["target"].cpu()).save(out / f"{split}_01_label_mIHC.png")
    print(f"[{split}] n={len(ds)}")
    print(f"  source: {b['source_path']}")
    print(f"  target: {b['target_path']}")
    print(f"  source tensor mean={b['source'].mean():.3f} (expect ~0.3-0.6 for H&E)")
    print(f"  target tensor mean={b['target'].mean():.3f} (often lower for dark mIHC)")
print(f"Wrote PNGs under {out}/")
print("Open *_00_input_HE.png — should look like pink/purple H&E.")
print("Open *_01_label_mIHC.png — should look like dark multiplex (green/red spots), NOT pink H&E.")
PY
