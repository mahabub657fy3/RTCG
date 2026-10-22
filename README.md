<div align="center">

# Go with the Flow: Generative Targeted Attacks via Restoration Transformers

**RTCG** — **R**estoration **T**ransformer-based **C**onditional **G**enerator

Md Mahabubur Rahman, Hui Zhang, Anjie Peng, Hui Zeng

Southwest University of Science and Technology, Mianyang, China

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.4.1](https://img.shields.io/badge/pytorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)

[**Paper**](#citation) (coming soon) · [**Checkpoints**](https://drive.google.com/drive/folders/1A1-6uryXixJcQNWuI1w0afUL-jQTK0Bz?usp=drive_link) · [**Test data**](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)

</div>

---

## Overview

RTCG is a **multi-target generative attack**. A single generator crafts targeted, transferable adversarial perturbations for any of eight ImageNet target classes in one forward pass.

Existing multi-target generators tend to produce a nearly image-agnostic template for each target. RTCG reduces this *template bias* with two design choices:

- **Restoration-transformer bottleneck.** Eight restoration transformer blocks (RTBs) work at 1/16 resolution. Each block uses channel-wise attention (MDTA) and a gated feed-forward network (GDFN), as in [Restormer](https://github.com/swz30/Restormer). The bottleneck adapts to each image and keeps its spatial layout.
- **Persistent spatial conditioning.** The target is a 2D Stable Diffusion-2 latent. It modulates every block of the encoder, bottleneck, and decoder per pixel. The 8 convolutional residual blocks (CRBs) use **SpatialFiLM**. The 8 RTBs use **SpatialAdaLN-Zero**. This gives 24 modulation sites in total. All modulators are zero-initialized.

<p align="center">
  <img src="assets/rtcg_framework.png" width="92%" alt="RTCG generator architecture">
</p>

The perturbation is bounded by construction, so no clipping is needed during training:

$$\delta = \epsilon \cdot \big(G * \tanh(f_\theta(x, c))\big), \qquad \lVert\delta\rVert_\infty \le \epsilon,$$

where $G$ is a fixed 3×3 Gaussian kernel, $x$ is the clean image and $c$ is the target latent.

<p align="center">
  <img src="assets/feature_maps.png" width="80%" alt="Bottleneck features of CGNC, TGAF and RTCG">
  <br>
  <em>Intermediate features of three conditional generators for the same input image. RTCG keeps the layout of the input through its bottleneck while the target is injected at every block.</em>
</p>

## Main Results

Targeted attack success rate (ASR, %) on the 1,000-image NeurIPS 2017 dev set. Settings: ε = 16/255, 325k training images, averaged over the 8 target classes. Only black-box victims are shown. All baselines were retrained from their official code under the same setting. See the paper for C-GSP, LFAA, GAKer, robust victims, JPEG defense, and smaller budgets.

**Surrogate: ResNet-50**

| Method | Inc-v3 | Res-152 | DN-121 | GoogleNet | VGG-16 | ViT-B | ViF-S | Swin-T | **Avg.** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CGNC | 19.14 | 41.78 | 51.33 | 22.12 | 36.76 | 9.33 | 14.33 | 7.06 | 25.23 |
| TGAF | 21.33 | 49.76 | 56.38 | 28.88 | 41.52 | 3.70 | 11.62 | 4.55 | 27.22 |
| **RTCG** | **28.96** | **59.59** | **72.23** | **39.37** | **50.88** | **12.35** | **31.58** | **18.37** | **39.17** |

**Surrogate: Inception-v3**

| Method | Res-152 | DN-121 | GoogleNet | VGG-16 | Res-50 | ViT-B | ViF-S | Swin-T | **Avg.** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CGNC | 2.73 | 6.20 | 3.73 | 3.23 | 5.65 | 1.26 | 2.19 | 0.58 | 3.20 |
| TGAF | 13.30 | 17.90 | 21.20 | 11.01 | 14.43 | 5.09 | 13.17 | 3.01 | 12.39 |
| **RTCG** | **36.08** | **39.26** | **29.47** | **28.81** | **27.66** | **9.79** | **18.81** | **4.59** | **24.31** |

The generator has 26.4M parameters. Of these, 17.8M belong to the target-only conditioning branch, and the image branch has 8.59M. At 224×224 it needs 26.0 GFLOPs, which is 3.8× fewer than TGAF.

## Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Quick Start](#quick-start)
- [Training](#training)
- [Generating Adversarial Examples](#generating-adversarial-examples)
- [Evaluating Targeted ASR](#evaluating-targeted-asr)
- [Target Classes and Conditioning Latents](#target-classes-and-conditioning-latents)
- [Reproducing the Paper](#reproducing-the-paper)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Repository Structure

```
RTCG/
├── models/
│   ├── generator.py                # RTCGenerator: encoder → 8 RTBs → decoder (U-Net)
│   ├── restoration_transformer.py  # RTB (MDTA + GDFN) and conditioned residual block (CRB)
│   └── spatial_modulation.py       # CondEncoder, SpatialFiLM, SpatialAdaLN-Zero
├── train.py              # Train a multi-target generator against a surrogate
├── eval.py               # Generate adversarial examples on the NeurIPS'17 dev set (all 8 targets)
├── inference.py          # Measure targeted ASR of the generated images on victim models
├── inference_jepg.py     # Same as inference.py, with a JPEG-compression defense
├── get_latent.py         # (Optional) Regenerate the class latents with Stable Diffusion 2
├── latents.pth           # Precomputed SD-2 latents for the 8 target classes
├── image_transformer.py  # Training augmentations (two-view transform, random 90° rotation)
├── utils.py              # Data loading, label mapping, target-class sets, victim model zoo
└── environment.yaml      # Conda environment used for the paper
```

> **Note on file names.** `eval.py` *generates* adversarial images. `inference.py` *evaluates* them.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/RTCG.git
cd RTCG
```

**Option A — exact paper environment (conda).**

```bash
conda env create -f environment.yaml
conda activate RTCG
```

**Option B — minimal install (pip).** This installs only the packages that RTCG imports.

```bash
conda create -n rtcg python=3.11 -y
conda activate rtcg
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install timm==1.0.11 einops==0.8.0 numpy pandas pillow tqdm

# Only needed to regenerate latents.pth with get_latent.py
pip install diffusers==0.30.3 transformers==4.44.2 accelerate==0.34.2 matplotlib
```

Training requires a CUDA GPU. Generation and evaluation fall back to the CPU, but they are slow there.

## Data Preparation

### Training data: ImageNet-1K

Use the ImageNet-1K (ILSVRC-2012) training set in the standard `ImageFolder` layout, with one subfolder per class:

```
/path/to/imagenet/train/
├── n01440764/
│   ├── n01440764_10026.JPEG
│   └── ...
└── ...
```

By default, `train.py` draws a fixed random subset of **325,000 images** (seed `1111`), as in the paper. Pass `--train_subset_size 0` to train on all images.

### Test data: NeurIPS 2017 dev set

Download the 1,000 test images and their metadata CSV from the [Kaggle competition page](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack). You must sign in and accept the competition rules first. With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle competitions download -c nips-2017-non-targeted-adversarial-attack
```

Arrange the files exactly like this:

```
data/nips2017_dev/
├── images.csv      # must contain the columns ImageId, TrueLabel, TargetClass
└── images/
    ├── <ImageId>.png
    └── ...         # 1,000 PNG images, 299×299
```

- `eval.py` reads this folder with `torchvision.datasets.ImageFolder`. Therefore `images/` must be the **only** subfolder inside `data/nips2017_dev/`.
- Labels in `images.csv` are 1-indexed (1–1000). The code shifts them to 0–999.
- If your copy names the metadata file differently (for example `dev_dataset.csv`), rename it to `images.csv`.

## Pretrained Checkpoints

Download the trained generators from **[Google Drive](https://drive.google.com/drive/folders/1A1-6uryXixJcQNWuI1w0afUL-jQTK0Bz?usp=drive_link)**. You can also fetch the whole folder from the command line:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1A1-6uryXixJcQNWuI1w0afUL-jQTK0Bz" -O checkpoints
```

The commands in this README assume this layout:

```
checkpoints/
├── res50/model-9.pth   # surrogate: ResNet-50,    input 224×224
└── incv3/model-9.pth   # surrogate: Inception-v3, input 299×299
```

Each file is a plain `state_dict` of `RTCGenerator`. Both surrogates use the same constructor arguments.

<details>
<summary><b>Use the generator from Python</b></summary>

```python
import torch
from models.generator import RTCGenerator

netG = RTCGenerator(cond_channels=4, cond_dim=128, dim=64, n_latent=8, n_heads_latent=8)
netG.load_state_dict(torch.load("checkpoints/res50/model-9.pth", map_location="cpu"))
netG.eval()

latents = torch.load("latents.pth", map_location="cpu")   # {class_id: tensor[4, 64, 64]}

x = torch.rand(4, 3, 224, 224)                    # images in [0, 1], NOT mean/std-normalized
cond = latents[150].unsqueeze(0).expand(x.size(0), -1, -1, -1)   # target 150: sea lion
eps = 16 / 255

with torch.no_grad():
    delta = netG(x, cond, eps=eps)                # ||delta||_inf <= eps
x_adv = (x + delta).clamp(0, 1)
```

</details>

## Quick Start

Evaluate the pretrained ResNet-50 generator in two steps:

```bash
# 1) Generate adversarial examples for all 8 targets (8 × 1,000 images)
python eval.py --model_type res50 \
               --load_path checkpoints/res50/model-9.pth \
               --data_dir data/nips2017_dev \
               --save_dir results/res50

# 2) Measure targeted ASR on the normally trained victims
python inference.py --test_dir results/res50/gan_n8/model-9 --model_t normal
```

For the Inception-v3 generator, use `--model_type incv3`, `--load_path checkpoints/incv3/model-9.pth` and `--save_dir results/incv3`.

> **Windows:** replace the `\` line continuations with `^` (cmd) or `` ` `` (PowerShell), or write each command on one line.

## Training

```bash
# ResNet-50 surrogate (224×224)
python train.py --train_dir /path/to/imagenet/train --model_type res50 --save_dir checkpoints/

# Inception-v3 surrogate (299×299)
python train.py --train_dir /path/to/imagenet/train --model_type incv3 --save_dir checkpoints/
```

Always pass `--train_dir`. Its default value is a local path from our machine.

| Argument | Default | Description |
|:--|:--|:--|
| `--train_dir` | — | ImageNet training folder (`ImageFolder` layout) |
| `--model_type` | `res50` | Surrogate: `res50` (224×224) or `incv3` (299×299) |
| `--eps` | `16` | L∞ budget in pixel units; ε = `eps`/255 |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Adam learning rate, β = (0.5, 0.999) |
| `--train_subset_size` | `325000` | Size of the random training subset; `0` uses all images |
| `--seed` | `1111` | Seed for the subset selection |
| `--label_flag` | `N8` | Target-class set (see [below](#target-classes-and-conditioning-latents)) |
| `--save_dir` | `checkpoints/` | Output root |
| `--start_epoch`, `--load_path` | `0`, `''` | Resume from a saved checkpoint |

**Objective.** The surrogate $F$ is frozen. Each image gets a target $t$ drawn uniformly from the eight classes. The generator is trained with cross-entropy towards $t$ on three views of each image:

$$\mathcal{L} = \mathrm{CE}\big(F(x+\delta), t\big) + \mathrm{CE}\big(F(x_{r}+\delta_{r}), t\big) + \mathrm{CE}\big(F(x_{a}+\delta_{a}), t\big),$$

where $x_r$ is $x$ rotated by a random multiple of 90°, and $x_a$ is a strong augmentation of $x$ (random resized crop, flip, color jitter, grayscale). Each perturbation is generated from its own view.

**Outputs.** A checkpoint is saved after every epoch as `checkpoints/<model_type>/model-<epoch>.pth`. The loss is logged every 100 iterations to `checkpoints/<model_type>/loss_log.txt`. The paper reports the final checkpoint, `model-9.pth`.

**Resuming.** For example, `--start_epoch 5 --load_path checkpoints/res50/model-4.pth`. The optimizer state is not saved, so Adam restarts from scratch.

**Cost.** With the ResNet-50 surrogate, batch size 8 and 325k images, training needs about 8 GB of GPU memory and about 3.2 hours per epoch. The Inception-v3 surrogate uses 299×299 inputs and needs more memory.

## Generating Adversarial Examples

`eval.py` crafts adversarial versions of all 1,000 test images for every target class.

| Argument | Default | Description |
|:--|:--|:--|
| `--data_dir` | — | NeurIPS'17 dev-set folder (see [Data Preparation](#data-preparation)) |
| `--model_type` | `incv3` | Sets the image size: `incv3` → 299, `res50` → 224. Must match the checkpoint's surrogate |
| `--load_path` | — | Generator checkpoint |
| `--save_dir` | `results/ccck` | Output root |
| `--eps` | `16` | L∞ budget in pixel units |
| `--batch_size` | `8` | Batch size |
| `--label_flag` | `N8` | Target-class set |

The adversarial images are projected onto the ε-ball, clamped to [0, 1], and saved as lossless PNG files:

```
results/res50/gan_n8/
├── model-9_t150/images/150_0.png … 150_999.png
├── model-9_t426/images/…
└── …                                  # one folder per target class
```

The folder prefix (`model-9`) comes from the checkpoint file name. Use a separate `--save_dir` for each checkpoint, so that two `model-9.pth` files do not overwrite each other's outputs.

## Evaluating Targeted ASR

`inference.py` counts how many adversarial images each victim classifies as the intended target. It then averages this rate over the eight targets.

```bash
python inference.py --test_dir results/res50/gan_n8/model-9 --model_t normal
```

Pass `--test_dir` **without** the `_t<class>` suffix. The script appends it for each target.

| `--model_t` | Victim models |
|:--|:--|
| `normal` | `incv3`, `vgg16`, `googlenet`, `res50`, `res152`, `dense121` |
| `vit` | `vit_base_patch16_224`, `pit_b_224`, `visformer_small`, `swin_tiny_patch4_window7_224` |
| `robust` | `adv_incv3`, `ens_inc_res_v2`, `res50_sin`, `res50_sin_in`, `res50_sin_fine_in`, `res50_augmix` |
| `all` | `vgg16`, `googlenet`, `incv3`, `res152`, `dense121`, `incv4`, `inc_res_v2`, `adv_incv3`, `ens_inc_res_v2`, `res50_sin`, `res50_sin_in`, `res50_sin_fine_in` |
| *single name* | Any key above, or `vgg19`, `dense201`, `incv4`, `inc_res_v2` |

The script prints the per-target success rates and a final dictionary `{victim: ASR}`. The surrogate's own column is a white-box result.

Torchvision and timm weights download automatically on first use. This includes `adv_incv3` and `ens_inc_res_v2`.

### Robust victims (manual download)

The Stylized-ImageNet and AugMix ResNet-50 models must be downloaded by hand:

| Key | Name | Source | File |
|:--|:--|:--|:--|
| `res50_sin` | Res-50<sub>SIN</sub> | [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) | `resnet50_train_60_epochs-c8e5653e.pth.tar` |
| `res50_sin_in` | Res-50<sub>IN</sub> | [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) | `resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar` |
| `res50_sin_fine_in` | Res-50<sub>FINE</sub> | [texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape) | `resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar` |
| `res50_augmix` | Res-50<sub>AugMix</sub> | [AugMix](https://github.com/google-research/augmix) | `checkpoint.pth.tar` |

Then edit the paths in `load_robust_model()` in `utils.py`, for example:

```python
local_weights = {
    'res50_sin':         'pretrained/resnet50_train_60_epochs-c8e5653e.pth.tar',
    'res50_sin_in':      'pretrained/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
    'res50_sin_fine_in': 'pretrained/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    'res50_augmix':      'pretrained/checkpoint.pth.tar',
}
```

These four models are loaded with `DataParallel` on the GPU, so they require CUDA.

### JPEG-compression defense

```bash
python inference_jepg.py --test_dir results/res50/gan_n8/model-9 --model_t normal --jpeg_quality 70
```

Each adversarial image is JPEG-compressed at quality `--jpeg_quality` before classification. The paper uses Q = 70, 80 and 90. Logs are written to `mydefense/jpeg/jepglog/`, and a JSON summary to `mydefense/jpeg/results/`.

## Target Classes and Conditioning Latents

RTCG follows the 8-target protocol of C-GSP and TGAF (`--label_flag N8`):

| ID | Class | ID | Class |
|--:|:--|--:|:--|
| 150 | sea lion | 952 | fig |
| 426 | barometer | 507 | combination lock |
| 843 | swing | 590 | hand-held computer |
| 715 | pickelhaube | 62 | rock python |

`latents.pth` holds one Stable Diffusion-2 latent per target class, as a dictionary `{class_id: tensor[4, 64, 64]}`. **All results in the paper use this file.** You do not need to regenerate it.

`get_latent.py` shows how the file was made:

- It uses Stable Diffusion 2 (base), DDIM with 25 steps, guidance scale 7.5, seed 0, and a class-specific prompt.
- It uses rejection sampling. A latent is kept only when ResNet-152 classifies its decoded image as the target class.
- It stores the latent after the 1/0.18215 VAE scaling. Accepted images are saved to `selected_images/`.

<details>
<summary><b>Regenerating the latents (optional)</b></summary>

1. Download the diffusion model:
   ```bash
   huggingface-cli download Manojb/stable-diffusion-2-base
   ```
2. Set `MODEL_CACHE_DIR` at the top of `get_latent.py` to the downloaded folder. On Linux this is usually `~/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-base`.
3. Back up the original `latents.pth`, then run:
   ```bash
   python get_latent.py
   ```

New latents may differ from the released ones across hardware and library versions. This can change the ASR slightly.

`utils.get_classes()` also defines the larger target sets `C20`, `C50`, `C100` and `C200`. The released `latents.pth` covers only `N8`. To use another set, add a prompt and a label for each class to `get_latent.py` and regenerate the file.

</details>

## Reproducing the Paper

| Experiment | How to run |
|:--|:--|
| Normally trained victims (CNNs) | `inference.py --model_t normal` |
| Vision transformers | `inference.py --model_t vit` |
| Robust and distribution-shifted victims | `inference.py --model_t robust` (after the [manual download](#robust-victims-manual-download)) |
| JPEG defense | `inference_jepg.py --jpeg_quality {70,80,90}` |
| Smaller budgets (ε = 8/255, 12/255) | Retrain with `train.py --eps {8,12}`, then generate with the same `--eps` |
| Training-set size (50k / 100k / 325k) | `train.py --train_subset_size {50000,100000,325000}` |

Run every experiment for both surrogates (`--model_type res50` and `--model_type incv3`). The ablation variants and the image-specificity diagnostics are not part of this release.

## Troubleshooting

- **Run all scripts from the repository root.** `latents.pth` is loaded by a relative path.
- **`KeyError` when looking up a latent.** You used a `--label_flag` other than `N8`. See [Target Classes](#target-classes-and-conditioning-latents).
- **`RuntimeError: ... filenames not found in images.csv`.** `--data_dir` must point to the folder that contains both `images.csv` and `images/`.
- **CUDA out of memory.** Lower `--batch_size`. This option exists in every script.
- **DataLoader errors on Windows.** Lower `num_workers` in the `DataLoader` calls (6 in `train.py`, 4 elsewhere).
- **Run-to-run variation.** Training is not bit-for-bit deterministic, because target sampling and augmentations are not seeded. Expect small differences between runs.
- **No internet on the compute node.** Pre-download the torchvision and timm victim weights on a machine with internet access, then copy the cache.

## Citation

If you find this work useful, please cite:

```bibtex
@article{rahman2026rtcg,
  title  = {Go with the Flow: Generative Targeted Attacks via Restoration Transformers},
  author = {Rahman, Md Mahabubur and Zhang, Hui and Peng, Anjie and Zeng, Hui},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgements

This code builds on the public implementations and evaluation protocols of [C-GSP](https://arxiv.org/abs/2107.01809), [CGNC](https://arxiv.org/abs/2407.10179) and [TGAF](https://arxiv.org/abs/2504.14137). The bottleneck blocks follow [Restormer](https://github.com/swz30/Restormer). The spatial modulators draw on [SPADE](https://github.com/NVlabs/SPADE) and the adaLN-Zero design of [DiT](https://github.com/facebookresearch/DiT). We thank the authors for releasing their code.

## Responsible Use

This repository is released for research on the robustness of image classifiers. Please use it only to evaluate and improve defenses, and only on systems you are authorized to test.
