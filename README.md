# Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model

**AAAI 2026 Accepted**

This repository provides the implementation of:

> **Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model**  
> Hyun-Jic Oh, Junsik Kim, Zhiyi Shi, Yichen Wu, Yu-An Chen, Peter K. Sorger, Hanspeter Pfister, Won-Ki Jeong

The code trains a marker-wise conditioned Diffusion-FT model for virtual staining from histology images. The current release includes training code for the ORION-CRC multiplex IF setting and the HEMIT paired translation setting.

## Overview

![Overview Figure](Figure/overview.png)

DiffVS follows a two-stage training recipe. Stage 1 performs Marigold-style paired latent diffusion training: the denoiser receives the noisy target-stain latent concatenated with the source-image latent, while a learnable marker token specifies the requested marker or stain. Stage 2 applies Diffusion-FT to the Stage-1 checkpoint for efficient one-step virtual staining.

## Repository Structure

```text
DiffVS/
  Figure/
    overview.png
  configs/
    orion_markers.txt
  scripts/
    train_orion_stage1_marigold.sh
    train_orion_stage2_diffusion_ft.sh
    train_hemit_stage1_marigold.sh
    train_hemit_stage2_diffusion_ft.sh
    infer_orion_diffusion_ft.sh
    infer_hemit_diffusion_ft.sh
  src/diffvs/
    datasets.py
    modeling.py
    train_stage1_marigold.py
    train_stage2_diffusion_ft.py
    infer_diffusion_ft.py
  requirements.txt
```

## Installation

```bash
git clone https://github.com/hvcl/DiffVS.git
cd DiffVS

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The training scripts use Hugging Face models (SD 2.1 base: VAE + UNet + scheduler).

**Note:** `stabilityai/stable-diffusion-2-1-base` was removed from Hugging Face in late 2025. Scripts default to the community mirror `Manojb/stable-diffusion-2-1-base`. Override with:

```bash
PRETRAINED_MODEL=Manojb/stable-diffusion-2-1-base   # or a local folder (see below)
```

Download once on a machine with internet, then copy to the cluster:

```bash
huggingface-cli download Manojb/stable-diffusion-2-1-base --local-dir ./pretrained/sd21-base
# on cluster:
PRETRAINED_MODEL=/path/to/pretrained/sd21-base bash scripts/train_hemit_stage1_marigold.sh
```

If a mirror is gated, log in:

```bash
huggingface-cli login
```

## Data Layout

All paths below are examples. Replace them with your local dataset locations.

### ORION-CRC

Expected layout:

```text
/path/to/ORIONCRC_dataset_tile_20x/
  train_dataframe.csv
  val_dataframe.csv
  test_dataframe.csv
  he/
  if/
```

The dataframe files must contain at least:

```text
image_path,target_path
```

`image_path` points to the H&E tile and `target_path` points to the multiplex IF TIFF tile. Relative paths are resolved under `DATASET_ROOT`.

### HEMIT

Expected layout (default: `DiffVS/data/` — set `DATASET_ROOT` if yours lives elsewhere):

```text
data/
  train/
    input/
    label/
  val/
    input/
    label/
  test/
    input/
    label/
```

The HEMIT loader treats the dataset as a single target-domain virtual staining task and uses one learnable target token named `HEMIT`.

## Training

The released code mirrors the two-stage procedure in the paper:

1. **Stage 1: Marigold-style training.** Train a paired latent diffusion model conditioned on the source image latent and the marker token.
2. **Stage 2: Diffusion-FT.** Initialize from the Stage-1 checkpoint and fine-tune at the one-step denoising timestep used for efficient inference.

### ORION-CRC

Stage 1:

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
AUGMENTED_DIR=/path/to/ORIONCRC_tile_20x_he_norm \
OUTPUT_DIR=./outputs/orion_stage1_marigold \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=15 \
bash scripts/train_orion_stage1_marigold.sh
```

Stage 2:

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
AUGMENTED_DIR=/path/to/ORIONCRC_tile_20x_he_norm \
STAGE1_CHECKPOINT_DIR=./outputs/orion_stage1_marigold/stage1-checkpoint-epoch-15 \
OUTPUT_DIR=./outputs/orion_stage2_diffusion_ft \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=5 \
bash scripts/train_orion_stage2_diffusion_ft.sh
```

To train a subset of markers:

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
OUTPUT_DIR=./outputs/orion_stage1_panck_sma \
bash scripts/train_orion_stage1_marigold.sh \
  --markers Hoechst Pan-CK SMA
```

### HEMIT

Put data under `DiffVS/data/{train,val,test}/{input,label}/` (or set `DATASET_ROOT`).

**Stage 1** — interactive:

```bash
OUTPUT_DIR=./outputs/hemit_stage1_marigold \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=100 \
bash scripts/train_hemit_stage1_marigold.sh
```

**Stage 1** — SLURM (same env vars as above):

```bash
cd /path/to/DiffVS
sbatch slurm/train_hemit_stage1_marigold.sbatch
```

**Stage 2** — interactive:

```bash
STAGE1_CHECKPOINT_DIR=./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100 \
OUTPUT_DIR=./outputs/hemit_stage2_diffusion_ft \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=5 \
bash scripts/train_hemit_stage2_diffusion_ft.sh
```

**Stage 2** — SLURM:

```bash
sbatch slurm/train_hemit_stage2_diffusion_ft.sbatch
```

## Inference

### ORION-CRC

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
CHECKPOINT_DIR=./outputs/orion_stage2_diffusion_ft/stage2-checkpoint-epoch-5 \
OUTPUT_DIR=./outputs/orion_inference \
bash scripts/infer_orion_diffusion_ft.sh
```

### HEMIT

**Inference** — interactive:

```bash
CHECKPOINT_DIR=./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5 \
OUTPUT_DIR=./outputs/hemit_inference \
bash scripts/infer_hemit_diffusion_ft.sh
```

**Inference** — SLURM:

```bash
sbatch slurm/infer_hemit_diffusion_ft.sbatch
```

Optional: `bash scripts/infer_hemit_diffusion_ft.sh --max_rows 32` for a quick smoke test.

**Troubleshooting garbage / rainbow panels**

The released `hvcl/DiffVS` `infer_diffusion_ft.py` uses **DDIM + random noise**, but **stage-2** trains with **DDPM @ t=999** only. That mismatch produces noise-like images even when training succeeded.

This fork’s `infer_diffusion_ft.py` auto-fixes stage-2 checkpoints (DDPM one-step). Confirm in the log:

```text
mode=stage2_ddpm_t999
```

If you still see `upstream_ddim_1step`, pull the latest code. Quick check:

```bash
bash scripts/verify_diffvs_checkpoint.sh
# or SLURM:
sbatch slurm/verify_diffvs_checkpoint.sbatch
```

Also verify `pretrained_model` in `outputs/hemit_stage2_diffusion_ft/config.json` matches what you used for training (e.g. `Manojb/stable-diffusion-2-1-base`).

**Metrics** (dual-branch `post_process.py`, comparable `score.csv`):

```bash
export PIX2PIX_ROOT=/path/to/Pix2pix_DualBranch
bash scripts/eval_hemit_metrics.sh
```

**Metrics** — SLURM:

```bash
export PIX2PIX_ROOT=/path/to/Pix2pix_DualBranch
sbatch slurm/eval_hemit_metrics.sbatch
```

## Checkpoints

Training writes checkpoints to `OUTPUT_DIR`:

```text
stage1-checkpoint-epoch-*/
stage2-checkpoint-epoch-*/
config.json
logs/
```

Each checkpoint stores:

```text
unet/
marker_encoder.pt
markers
```

Large checkpoints and generated outputs are ignored by git.

## Citation

If this code or paper is useful for your research, please cite:

```bibtex
@article{oh2025virtual,
  title   = {Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model},
  author  = {Oh, Hyun-Jic and Kim, Junsik and Shi, Zhiyi and Wu, Yichen and Chen, Yu-An and Sorger, Peter K. and Pfister, Hanspeter and Jeong, Won-Ki},
  journal = {arXiv preprint arXiv:2508.14681},
  year    = {2025}
}
```

## Acknowledgements

This implementation builds on ideas and components from several excellent open-source projects. In particular, we thank the authors of:

- [Marigold](https://github.com/prs-eth/Marigold)
- [diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft)

Please also cite those projects when using their models, code, or training recipes.

## License

Please check the licenses of this repository and the referenced pretrained models before commercial use. Dataset access and redistribution are governed by the original dataset providers.
