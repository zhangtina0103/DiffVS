from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
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
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Cap number of tiles (default: all). Use e.g. 32 for a quick smoke test.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="Manojb/stable-diffusion-2-1-base",
        help="HF repo id or local dir with vae/, unet/, scheduler/",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def encode_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    images = images * 2.0 - 1.0
    return vae.encode(images).latent_dist.mode() * vae.config.scaling_factor


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latents / vae.config.scaling_factor).sample
    return (images / 2.0 + 0.5).clamp(0, 1)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(tensor.detach().cpu()).save(path)


def make_panel(source: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, path: Path) -> None:
    width, height = source.shape[-1], source.shape[-2]
    panel = Image.new("RGB", (width * 3, height))
    panel.paste(to_pil_image(source.detach().cpu()), (0, 0))
    panel.paste(to_pil_image(target.detach().cpu()), (width, 0))
    panel.paste(to_pil_image(pred.detach().cpu()), (width * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_ckpt = torch.load(Path(args.checkpoint_dir) / "marker_encoder.pt", map_location="cpu")
    ckpt_markers = list(marker_ckpt.get(
        "markers",
        DEFAULT_ORION_MARKERS if args.dataset == "orion" else ["HEMIT"],
    ))
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

    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(Path(args.checkpoint_dir) / "unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    marker_encoder = MarkerTokenEncoder(
        marker_names=ckpt_markers,
        cross_attention_dim=int(marker_ckpt["cross_attention_dim"]),
    )
    marker_encoder.load_state_dict(marker_ckpt["state_dict"])
    marker_encoder.to(device)
    marker_encoder.eval()
    vae.eval()
    unet.eval()

    marker_to_id = {name: idx for idx, name in enumerate(ckpt_markers)}
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="DiffVS inference")):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            marker_name_batch = batch["marker_name"]
            marker_ids = torch.tensor([marker_to_id[str(name)] for name in marker_name_batch], device=device)
            source_latents = encode_latents(vae, source)
            latents = torch.randn_like(source_latents)
            context = marker_encoder(marker_ids)

            for t in scheduler.timesteps:
                model_input = torch.cat([latents, source_latents], dim=1)
                noise_pred = unet(model_input, t, encoder_hidden_states=context, return_dict=False)[0]
                latents = scheduler.step(noise_pred, t, latents, eta=args.eta).prev_sample

            pred = decode_latents(vae, latents)
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

    with open(output_dir / "inference_manifest.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
