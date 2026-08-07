# 🪨 RockNet++ — Two Ways to Fix Skip Connections for Rock Grain Boundary Segmentation

> A U-Net / UNet++ ablation study exploring **two independent fixes** for weak skip connections: (A) **direct long-range shortcuts** that bypass intermediate nested convolutions, and (B) **trainable Sobel/Laplacian edge kernels** fused into every skip path.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Colab](https://img.shields.io/badge/Made%20in-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-blueviolet)]()

---

## 📌 Overview

This repository documents an applied research project (Research Internship) on **semantic segmentation of rock grain boundaries** from petrographic/microscopy images — used in geology and materials science to automatically measure grain size, shape, and distribution instead of manual tracing.

The project runs a structured **ablation study** across two backbones (**U-Net** and **UNet++**) and, for each, tests **two separate modifications to the skip connections**:

- **Modification A — Long-Range / Direct Skips**: add extra shortcut connections *on top of* the existing skip connections, so raw encoder features reach far decoder stages directly instead of only through several intermediate nested convolutions.
- **Modification B — Learnable Edge-Kernel Fusion**: fuse a trainable depthwise kernel (initialized with Sobel-X, Sobel-Y, and Laplacian filters) into each skip connection, so boundary/edge information is explicitly reinforced at every resolution.

## 🧠 Modification A — Long-Range / Direct Skip Connections

**Motivation:** in a plain U-Net, each decoder stage only receives its one matched skip. In UNet++, a raw encoder feature has to pass through *several* intermediate nested dense nodes before reaching a far decoder stage — and each hop dilutes high-level semantic information a little more.

**Fix:** add a direct shortcut from the raw encoder feature straight to a farther decoder node, **in addition to** all the normal skip connections (nothing is removed).

**U-Net version** (`Skip_connections_modified_unet_1.ipynb`) — each decoder stage fuses its own matched skip *and* the next-deeper one:

```
up1 (16→32)   = bottleneck            + c4
up2 (32→64)   = up1                   + c3 + c4   (c4 = long-range)
up3 (64→128)  = up2                   + c2 + c3   (c3 = long-range)
up4 (128→256) = up3                   + c1 + c2   (c2 = long-range)
```

**UNet++ version** (`Skip_connections_modified1_UNET++.ipynb`) — three explicit shortcuts added on top of the standard nested pathways:

```
X(1,0) ──────────────► X(0,4)   (skips 3 intermediate nested hops)
X(2,0) ──────────────► X(1,3)
X(3,0) ──────────────► X(2,2)
```

Each shortcut is implemented as its own `Conv2DTranspose` upsample of the raw encoder node, concatenated into that decoder node alongside its usual nested inputs.

## 🧠 Modification B — Learnable Edge-Kernel Fusion

**Motivation:** rather than hoping the network learns to detect grain boundaries from scratch, give it a classical-CV edge-detection prior — and let gradient descent fine-tune it.

**Fix:** a custom Keras layer (`LearnableDepthwiseConv2D`) whose kernel is *initialized* with Sobel-X, Sobel-Y, and Laplacian filters, then set `trainable=True`. Its response is concatenated with the original feature and compressed back down with a 1×1 conv, for every skip connection it's applied to:

```
skip feature ──┬──────────────────────────────┐
               │                                │
               └── LearnableDepthwiseConv2D     │
                   (init: Sobel-X/Y, Laplacian) │
                   (trainable)                  │
                        │                       │
                        ▼                       ▼
                     Concatenate ──► 1×1 Conv (compress) ──► conv block
```

Applied to plain U-Net skips in `Skip_connections_modified_unet_2/3.ipynb`, and to *every nested node* of UNet++ (combined with deep supervision) in `Skip_Connections_modified2/3_UNET++.ipynb` — with v3 being a refined implementation (`@register_keras_serializable`, added BatchNorm in the kernel branch).

## 📊 Dataset

- Paired **image / binary mask** dataset of rock-grain micrographs, resized to `256×256`
- Small, expert-annotated pilot set (12 train / 4 validation images) — reflects the real constraint in geoscience imaging, where pixel-accurate grain masks are expensive to produce
- On-the-fly augmentation applied during training via `tf.data`
- Loaded and paired automatically by filename stem (`get_pairs()`)

## 📈 Results

Best validation checkpoint per notebook (monitored on `val_dice_coef` / `val_avg_dice_coef`):

| # | Model | Modification | Val Dice ↑ | Val IoU ↑ |
|---|---|---|---|---|
| 1 | `Baseline_model.ipynb` | Vanilla U-Net | 0.696 | 0.534 |
| 2 | `Skip_connections_modified_unet_1.ipynb` | U-Net + long-range skips | 0.748 | 0.598 |
| 3 | `Skip_connections_modified_unet_2.ipynb` | U-Net + edge-kernel skips | 0.705 | 0.544 |
| 4 | `Skip_connections_modified_unet_3.ipynb` | U-Net + edge-kernel skips (refined) | 0.687 | 0.523 |
| 5 | `Baseline_UNET++.ipynb` | Paper-faithful UNet++ | 0.740 | 0.587 |
| 6 | `Skip_connections_modified1_UNET++.ipynb` | UNet++ + long-range skips | 0.741 | 0.589 |
| 7 | `Skip_Connections_modified2_UNET++.ipynb` | **UNet++ + edge-kernel skips** | **0.762** | **0.616** |
| 8 | `Skip_Connections_modified3_UNET++.ipynb` | UNet++ + edge-kernel skips (refined) | 0.750 | 0.600 |

🏆 **Best model: #7 — UNet++ + Learnable Edge-Kernel Fusion.** Dice **+6.6 points** and IoU **+8.2 points** over the vanilla U-Net baseline; Dice +2.2 / IoU +2.9 over the paper-faithful UNet++ baseline.

**Two honest observations from the ablation:**
- On the plain U-Net backbone, the **long-range skip** modification (#2) actually beat both edge-kernel variants (#3, #4) — nested architecture wasn't required to benefit from direct shortcuts.
- The "refined" edge-kernel pass (v3) scored slightly *lower* than v2 on both backbones (#4 < #3, #8 < #7) — a useful negative result showing the refinement (added BatchNorm in the kernel branch) didn't pay off on this small dataset.

*(Numbers will shift run-to-run given the tiny 4-image validation set — treat as directional, not production SLAs.)*

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `OpenCV` · `NumPy` · `scikit-learn` (train/val split) · `Matplotlib` (training curves & prediction viz) · `Google Colab` (GPU training)

**Techniques used:** U-Net & UNet++ architectures · nested/dense skip pathways · long-range/direct skip shortcuts · deep supervision with averaged multi-head inference · custom Keras layers (`LearnableDepthwiseConv2D`) · classical CV kernel initialization (Sobel, Laplacian) as a learnable prior · Dice loss / combined BCE+Dice loss · Dice & IoU metrics · `tf.data` pipelines with augmentation · model checkpointing, early stopping, LR scheduling

## 📂 Repository Structure

```
Rock-segmentation/
├── Baseline_model.ipynb                       # Vanilla U-Net baseline
├── Baseline_UNET++.ipynb                       # Paper-faithful UNet++ (no modifications)
│
├── Skip_connections_modified_unet_1.ipynb      # U-Net  + long-range skip shortcuts
├── Skip_connections_modified_unet_2.ipynb      # U-Net  + Sobel/Laplacian edge-kernel skips
├── Skip_connections_modified_unet_3.ipynb      # U-Net  + edge-kernel skips (refined)
│
├── Skip_connections_modified1_UNET++.ipynb     # UNet++ + long-range skip shortcuts
├── Skip_Connections_modified2_UNET++.ipynb     # UNet++ + Sobel/Laplacian edge-kernel skips (best)
├── Skip_Connections_modified3_UNET++.ipynb     # UNet++ + edge-kernel skips (refined)
└── README.md
```

## ▶️ Running It

All notebooks are self-contained and built to run on **Google Colab** with a mounted Google Drive dataset (`/content/drive/MyDrive/dataset_rock/{images,masks}`).

1. Open any notebook in Colab (each has an "Open in Colab" badge)
2. Mount your Drive and point `IMAGE_DIR` / `MASK_DIR` to your dataset
3. Run all cells — training curves (accuracy / Dice / IoU / loss) and side-by-side prediction visualizations are generated automatically
4. Best weights are checkpointed to `.keras` files based on validation Dice

## 🔭 Future Work

- Scale up the annotated dataset beyond 16 images to get statistically reliable metrics
- Combine both modifications: long-range skips **and** edge-kernel fusion in the same model
- Try a pretrained encoder backbone (e.g. VGG16/ResNet) for transfer learning
- Post-process predicted masks with watershed to separate touching grains into instances
- Package the best model (#7) behind a small inference API/UI for quick mask generation on new micrographs

## 👤 Author
**Prof. Anjan Chowdhury** — Assistant Professor, IIT (ISM) Dhanbad
**Karthikeya** — Student, IIT (ISM) Dhanbad
Research Intern project exploring computer vision & deep learning for geoscience image analysis.

---

*If you use or build on this work, a ⭐ on the repo is always appreciated.*
