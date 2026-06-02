from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

try:
    from .datasets import DEFAULT_ORION_MARKERS, build_dataset
    from .modeling import MarkerTokenEncoder
except ImportError:
    from datasets import DEFAULT_ORION_MARKERS, build_dataset
    from modeling import MarkerTokenEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run marker-wise conditioned Diffusion-FT inference.")
    parser.add_argument("--dataset", choices=["orion", "hemit"], required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--markers", nargs="+", default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="Must match training (default: read from config.json next to checkpoint).",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--force_upstream_infer",
        action="store_true",
        help="Use original hvcl DDIM+randn (broken for stage-2 FT checkpoints).",
    )
    return parser.parse_args()


def encode_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    images = images * 2.0 - 1.0
    return vae.encode(images).latent_dist.mode() * vae.config.scaling_factor


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latents / vae.config.scaling_factor).sample
    return (images / 2.0 + 0.5).clamp(0, 1)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(tensor.detach().cpu().clamp(0, 1)).save(path)


def make_panel(source: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, path: Path) -> None:
    width, height = source.shape[-1], source.shape[-2]
    panel = Image.new("RGB", (width * 3, height))
    panel.paste(to_pil_image(source.detach().cpu().clamp(0, 1)), (0, 0))
    panel.paste(to_pil_image(target.detach().cpu().clamp(0, 1)), (width, 0))
    panel.paste(to_pil_image(pred.detach().cpu().clamp(0, 1)), (width * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def load_training_config(checkpoint_dir: Path) -> dict:
    for parent in (checkpoint_dir, checkpoint_dir.parent):
        cfg_path = parent / "config.json"
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                return json.load(f)
    return {}


def is_stage2_checkpoint(cfg: dict, checkpoint_dir: Path) -> bool:
    if cfg.get("stage") == "stage2_diffusion_ft":
        return True
    return "stage2-checkpoint" in checkpoint_dir.name


def resolve_pretrained(requested: str, cfg: dict) -> str:
    if requested:
        return requested
    if cfg.get("pretrained_model"):
        return str(cfg["pretrained_model"])
    return "Manojb/stable-diffusion-2-1-base"


def infer_stage2_ddpm(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    source_latents: torch.Tensor,
    marker_context: torch.Tensor,
    timestep: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Matches train_stage2_diffusion_ft.py (DDPM @ t=999, noise on latents)."""
    t = torch.tensor([timestep], device=source_latents.device, dtype=torch.long)
    noise = torch.randn(
        source_latents.shape,
        device=source_latents.device,
        dtype=source_latents.dtype,
        generator=generator,
    )
    # Training uses add_noise(target_latents, ...); at test we use source_latents as x0 prior.
    latents = scheduler.add_noise(source_latents, noise, t)
    model_input = torch.cat([latents, source_latents], dim=1)
    noise_pred = unet(model_input, t, encoder_hidden_states=marker_context, return_dict=False)[0]
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    return decode_latents(vae, latents)


def infer_upstream_ddim(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    scheduler: DDIMScheduler,
    source_latents: torch.Tensor,
    marker_context: torch.Tensor,
    num_inference_steps: int,
    eta: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Original hvcl/DiffVS infer_diffusion_ft.py (DDIM + randn)."""
    latents = torch.randn(
        source_latents.shape,
        device=source_latents.device,
        dtype=source_latents.dtype,
        generator=generator,
    )
    scheduler.set_timesteps(num_inference_steps, device=source_latents.device)
    for t in scheduler.timesteps:
        model_input = torch.cat([latents, source_latents], dim=1)
        noise_pred = unet(model_input, t, encoder_hidden_states=marker_context, return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
    return decode_latents(vae, latents)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)

    cfg = load_training_config(checkpoint_dir)
    stage2 = is_stage2_checkpoint(cfg, checkpoint_dir)
    pretrained_model = resolve_pretrained(args.pretrained_model, cfg)
    ft_timestep = int(cfg.get("single_step_timestep", 999))

    marker_ckpt = torch.load(checkpoint_dir / "marker_encoder.pt", map_location="cpu")
    ckpt_markers = list(
        marker_ckpt.get(
            "markers",
            DEFAULT_ORION_MARKERS if args.dataset == "orion" else ["HEMIT"],
        )
    )
    marker_names = args.markers or ckpt_markers
    dataset, marker_names = build_dataset(
        dataset=args.dataset,
        root_dir=args.dataset_root,
        split=args.split,
        image_size=args.image_size,
        markers=marker_names,
        max_rows=args.max_rows,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device).float()
    unet = UNet2DConditionModel.from_pretrained(checkpoint_dir / "unet").to(device).float()
    marker_encoder = MarkerTokenEncoder(
        marker_names=ckpt_markers,
        cross_attention_dim=int(marker_ckpt["cross_attention_dim"]),
    )
    marker_encoder.load_state_dict(marker_ckpt["state_dict"])
    marker_encoder.to(device).float().eval()
    vae.eval()
    unet.eval()

    ddim_steps = args.num_inference_steps
    if not stage2 and not args.force_upstream_infer and ddim_steps == 1:
        ddim_steps = 25
        print("Stage-1 checkpoint: using 25 DDIM steps (official 1-step DDIM is not enough).")

    if stage2 and not args.force_upstream_infer:
        infer_mode = f"stage2_ddpm_t{ft_timestep}"
        print(
            "NOTE: hvcl release infer uses DDIM+randn, which does NOT match stage-2 training. "
            f"Using DDPM one-step @ t={ft_timestep} instead."
        )
    else:
        infer_mode = f"upstream_ddim_{ddim_steps}step"
        if stage2:
            print("WARN: --force_upstream_infer on stage-2 ckpt → expect garbage (repo mismatch).")

    print(f"pretrained_model={pretrained_model}")
    print(f"checkpoint={checkpoint_dir} mode={infer_mode}")

    marker_to_id = {name: idx for idx, name in enumerate(ckpt_markers)}
    rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="DiffVS inference")):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            marker_name_batch = batch["marker_name"]
            marker_ids = torch.tensor([marker_to_id[str(name)] for name in marker_name_batch], device=device)
            source_latents = encode_latents(vae, source)
            context = marker_encoder(marker_ids)

            if stage2 and not args.force_upstream_infer:
                scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
                pred = infer_stage2_ddpm(
                    unet, vae, scheduler, source_latents, context, ft_timestep, generator
                )
            else:
                scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
                pred = infer_upstream_ddim(
                    unet, vae, scheduler, source_latents, context, ddim_steps, args.eta, generator
                )

            for i in range(pred.shape[0]):
                marker = str(marker_name_batch[i])
                stem = f"{batch_idx:05d}_{i:02d}_{marker.replace('/', '-')}"
                pred_path = output_dir / "predictions" / f"{stem}.png"
                panel_path = output_dir / "panels" / f"{stem}_panel.png"
                save_tensor_image(pred[i], pred_path)
                make_panel(source[i], target[i], pred[i], panel_path)
                rows.append(
                    {
                        "marker": marker,
                        "source_path": str(batch["source_path"][i]),
                        "target_path": str(batch["target_path"][i]),
                        "prediction_path": str(pred_path),
                        "panel_path": str(panel_path),
                    }
                )

    meta = {
        "pretrained_model": pretrained_model,
        "checkpoint_dir": str(checkpoint_dir),
        "infer_mode": infer_mode,
        "training_stage": cfg.get("stage"),
    }
    with open(output_dir / "inference_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(output_dir / "inference_manifest.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
