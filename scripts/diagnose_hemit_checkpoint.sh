#!/usr/bin/env bash
# If stage-1 reached epoch 100 but inference is rainbow, run this on the cluster.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/data}"
CKPT="${CKPT:-${REPO_DIR}/outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
OUT="${OUT:-${REPO_DIR}/outputs/hemit_diagnose}"
PRETRAINED="${PRETRAINED:-}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
mkdir -p "${OUT}" logs

echo "=== HEMIT checkpoint diagnose ==="
echo "CKPT=${CKPT}"
echo "DATASET_ROOT=${DATASET_ROOT}"

if [[ -f "${REPO_DIR}/outputs/hemit_stage1_marigold/config.json" ]]; then
  echo "--- outputs/hemit_stage1_marigold/config.json ---"
  cat "${REPO_DIR}/outputs/hemit_stage1_marigold/config.json"
  echo ""
fi

if ls "${REPO_DIR}"/logs/diffvs_hemit_stage1_*.out 1>/dev/null 2>&1; then
  echo "--- last training losses (grep loss=) ---"
  grep -h "loss=" "${REPO_DIR}"/logs/diffvs_hemit_stage1_*.out 2>/dev/null | tail -15 || true
  echo ""
fi

python - <<PY
import json
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image

REPO = Path("${REPO_DIR}")
sys.path.insert(0, str(REPO / "src"))
from diffvs.datasets import build_dataset
from diffvs.infer_diffusion_ft import decode_latents, encode_latents, infer_upstream_ddim
from diffvs.modeling import MarkerTokenEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = Path("${CKPT}")
out = Path("${OUT}")
cfg_path = ckpt.parent / "config.json"
cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
pretrained = "${PRETRAINED}" or cfg.get("pretrained_model") or "Manojb/stable-diffusion-2-1-base"
print("pretrained_model:", pretrained)

ds, _ = build_dataset("hemit", "${DATASET_ROOT}", "test", 256, ["HEMIT"], max_rows=1)
b = ds[0]
src = b["source"].unsqueeze(0).to(device)
tgt = b["target"].unsqueeze(0).to(device)
to_pil_image(src[0].cpu()).save(out / "00_source.png")
to_pil_image(tgt[0].cpu()).save(out / "01_gt_label.png")

vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae").to(device).float()
vae.eval()
with torch.no_grad():
    rt = decode_latents(vae, encode_latents(vae, tgt))
to_pil_image(rt[0].cpu()).save(out / "02_vae_roundtrip_gt.png")
print("If 02 looks like rainbow but 01 is fine → VAE cannot represent mIHC (pretrained issue).")

unet = UNet2DConditionModel.from_pretrained(ckpt / "unet").to(device).float()
print("unet in_channels:", unet.config.in_channels, "(expect 8)")

mk = torch.load(ckpt / "marker_encoder.pt", map_location="cpu")
enc = MarkerTokenEncoder(list(mk["markers"]), int(mk["cross_attention_dim"]))
enc.load_state_dict(mk["state_dict"])
enc.to(device).float().eval()
ctx = enc(torch.zeros(1, dtype=torch.long, device=device))

src_lat = encode_latents(vae, src)
ddim = DDIMScheduler.from_pretrained(pretrained, subfolder="scheduler")
gen = torch.Generator(device=device).manual_seed(7)
for steps in (25, 50):
    g = torch.Generator(device=device).manual_seed(7)
    pred = infer_upstream_ddim(unet, vae, ddim, src_lat, ctx, steps, 0.0, g)
    to_pil_image(pred[0].cpu()).save(out / f"03_ddim_{steps}step.png")
    print(f"Wrote 03_ddim_{steps}step.png")

summary = {
    "pretrained": pretrained,
    "unet_in_channels": unet.config.in_channels,
    "dataset_root": "${DATASET_ROOT}",
    "checkpoint": str(ckpt),
    "interpret": {
        "02_bad_01_ok": "SD VAE bad for mIHC RGB — not fixable by more epochs alone",
        "02_ok_03_bad": "Training failed or wrong data; check loss~0.05-0.3 at end, audit data",
        "unet_not_8": "Checkpoint corrupt / wrong UNet saved",
    },
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo ""
echo "Open: ${OUT}/"
echo "  02_vae_roundtrip_gt.png  — VAE sanity"
echo "  03_ddim_25step.png       — your stage-1 infer path"
