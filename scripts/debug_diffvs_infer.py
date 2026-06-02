#!/usr/bin/env python3
"""One-tile diagnostic: which inference path works for your checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image

from diffvs.datasets import build_dataset
from diffvs.infer_diffusion_ft import (
    decode_latents,
    encode_latents,
    load_checkpoint_config,
    resolve_pretrained_model,
    run_ddim,
    run_stage2_ddpm,
)
from diffvs.modeling import MarkerTokenEncoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--pretrained_model", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./outputs/debug_infer")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_checkpoint_config(ckpt)
    pretrained = resolve_pretrained_model(args.pretrained_model, cfg)

    print("checkpoint:", ckpt)
    print("stage:", cfg.get("stage"))
    print("pretrained:", pretrained)

    ds, _ = build_dataset("hemit", args.dataset_root, "test", 256, ["HEMIT"], max_rows=1)
    b = ds[0]
    source = b["source"].unsqueeze(0).to(device)
    target = b["target"].unsqueeze(0).to(device)

    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae").to(device).float()
    unet = UNet2DConditionModel.from_pretrained(ckpt / "unet").to(device).float()
    print("unet in_channels:", unet.config.in_channels)

    mk = torch.load(ckpt / "marker_encoder.pt", map_location="cpu")
    enc = MarkerTokenEncoder(["HEMIT"], int(mk["cross_attention_dim"]))
    enc.load_state_dict(mk["state_dict"])
    enc.to(device).float().eval()
    ctx = enc(torch.zeros(1, dtype=torch.long, device=device))

    to_pil_image(source[0].cpu()).save(out / "00_source.png")
    to_pil_image(target[0].cpu()).save(out / "00_ground_truth.png")
    to_pil_image(decode_latents(vae, encode_latents(vae, target))[0].cpu()).save(
        out / "01_vae_roundtrip_gt.png"
    )

    src_lat = encode_latents(vae, source)
    tgt_lat = encode_latents(vae, target)
    gen = torch.Generator(device=device).manual_seed(7)
    ddpm = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
    ddim = DDIMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    to_pil_image(
        run_stage2_ddpm(unet, vae, ddpm, tgt_lat, src_lat, ctx, 999, "source", gen)[0].cpu()
    ).save(out / "02_ddpm_source_init.png")
    gen = torch.Generator(device=device).manual_seed(7)
    to_pil_image(
        run_stage2_ddpm(unet, vae, ddpm, tgt_lat, src_lat, ctx, 999, "target", gen)[0].cpu()
    ).save(out / "03_ddpm_target_init_ORACLE.png")
    gen = torch.Generator(device=device).manual_seed(7)
    to_pil_image(run_ddim(unet, vae, ddim, src_lat, ctx, 1, 0.0, gen)[0].cpu()).save(
        out / "04_ddim_1step.png"
    )
    gen = torch.Generator(device=device).manual_seed(7)
    to_pil_image(run_ddim(unet, vae, ddim, src_lat, ctx, 25, 0.0, gen)[0].cpu()).save(
        out / "05_ddim_25step.png"
    )

    summary = {
        "pretrained": pretrained,
        "checkpoint": str(ckpt),
        "unet_in_channels": unet.config.in_channels,
        "interpretation": {
            "01": "≈ GT → VAE/pretrained OK",
            "03": "≈ GT → stage2 trained OK; use ddpm+target init at train, try ddpm+source at test",
            "05": "good on stage1 ckpt → use DDIM 25 steps, not stage2 1-step",
            "all noise": "pretrained_model mismatch or bad checkpoint",
        },
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
