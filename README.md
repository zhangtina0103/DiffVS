# Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model

**AAAI 2026 Accepted**

This repository provides the implementation of:

> **Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model**  
> Hyun-Jic Oh, Junsik Kim, Zhiyi Shi, Yichen Wu, Yu-An Chen, Peter K. Sorger, Hanspeter Pfister, Won-Ki Jeong

The code trains a marker-wise conditioned Diffusion-FT model for virtual staining from histology images. The current release includes training code for the ORION-CRC multiplex IF setting and the HEMIT paired translation setting.

## Overview

![Overview Figure](Figure/overview.png)

DiffVS fine-tunes a latent diffusion denoiser for paired virtual staining. The denoiser receives the noisy target-stain latent concatenated with the source-image latent, while a learnable marker token specifies the requested marker or stain. This allows a single diffusion model to synthesize multiple target channels while sharing the same image-to-image backbone.

## Repository Structure

```text
DiffVS/
  Figure/
    overview.png
  configs/
    orion_markers.txt
  scripts/
    train_orion_diffusion_ft.sh
    train_hemit_diffusion_ft.sh
    infer_orion_diffusion_ft.sh
    infer_hemit_diffusion_ft.sh
  src/diffvs/
    datasets.py
    modeling.py
    train_diffusion_ft.py
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

The training scripts use Hugging Face models. Make sure you have access to the corresponding model repositories and are logged in if required:

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

Expected layout:

```text
/path/to/HEMIT/
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

### ORION-CRC

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
AUGMENTED_DIR=/path/to/ORIONCRC_tile_20x_he_norm \
OUTPUT_DIR=./outputs/orion_diffusion_ft \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=15 \
bash scripts/train_orion_diffusion_ft.sh
```

To train a subset of markers:

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
OUTPUT_DIR=./outputs/orion_panck_sma \
bash scripts/train_orion_diffusion_ft.sh \
  --markers Hoechst Pan-CK SMA
```

### HEMIT

```bash
DATASET_ROOT=/path/to/HEMIT \
OUTPUT_DIR=./outputs/hemit_diffusion_ft \
NUM_PROCESSES=1 \
TRAIN_BATCH_SIZE=16 \
NUM_EPOCHS=100 \
bash scripts/train_hemit_diffusion_ft.sh
```

## Inference

### ORION-CRC

```bash
DATASET_ROOT=/path/to/ORIONCRC_dataset_tile_20x \
CHECKPOINT_DIR=./outputs/orion_diffusion_ft/checkpoint-epoch-15 \
OUTPUT_DIR=./outputs/orion_inference \
bash scripts/infer_orion_diffusion_ft.sh
```

### HEMIT

```bash
DATASET_ROOT=/path/to/HEMIT \
CHECKPOINT_DIR=./outputs/hemit_diffusion_ft/checkpoint-epoch-100 \
OUTPUT_DIR=./outputs/hemit_inference \
bash scripts/infer_hemit_diffusion_ft.sh
```

Inference writes generated images, three-column panels, and an `inference_manifest.json`.

## Checkpoints

Training writes checkpoints to `OUTPUT_DIR`:

```text
checkpoint-epoch-*/
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
