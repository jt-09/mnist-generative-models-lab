# MNIST Generative Models Lab

A notebook-driven learning lab on MNIST that progresses from scratch implementations (KNN, logistic regression, MLP) through EDA and CNN classification to generative models: VAE, pixel-space DDPM, DCGAN, and VAE + latent diffusion MLP. Ten interactive notebooks preserve embedded outputs so metrics and plots are visible without retraining.

**Technical report:** [`reports/report.pdf`](reports/report.pdf) (PDF) | [`reports/report.md`](reports/report.md) (Markdown)  
**Collated figures and metrics:** [`results/RESULTS_INDEX.md`](results/RESULTS_INDEX.md)

## Introduction

I built this repo as a hands-on curriculum, not a single benchmark script. The goal is to make each modeling step visible: explore MNIST, implement classical classifiers without frameworks, add a small CNN, benchmark models side by side, then walk through four generative families (VAE, DDPM, DCGAN, latent diffusion). Every notebook keeps saved cell outputs, so you can read the results without retraining. A reflective technical report documents what I learned, including awkward outcomes like logistic regression underperforming KNN and soft VAE reconstructions.

## Project outline

| Phase | Focus | Key notebooks | Outcome |
|-------|--------|---------------|---------|
| 1. Setup and EDA | Data intuition, class balance, PCA | `eda_analysis.ipynb` | Motivation for non-linear models |
| 2. Manual classification | KNN, logistic regression, MLP from scratch | `manual_knn.ipynb`, `manual_logistic_reg.ipynb`, `manual_mlp.ipynb` | Accuracy ladder to **0.9794** (MLP) |
| 3. Deep learning baseline | PyTorch CNN | `cnn.ipynb` | **0.9871** test accuracy |
| 4. Benchmark | sklearn + PyTorch comparison harness | `02_mnist_models.ipynb` | CNN **0.9913**; confusion analysis |
| 5. Generative models | VAE, pixel DDPM, DCGAN, latent diffusion MLP | `vae.ipynb`, `ddpm.ipynb`, `dcgan.ipynb`, latent-diff notebook | Qualitative digit samples in `results/figures/` |

Two tracks run through the repo: **classification** (phases 1 to 4) and **generation** (phase 5). See the learning roadmap below for the recommended notebook order.

## Why this project

MNIST is the canonical entry point for machine learning. This repo treats it as a full curriculum: understand data (EDA), build classifiers without frameworks (manual notebooks), add deep learning (CNN), benchmark everything side by side, then explore generation (VAE → diffusion → GAN → latent diffusion). Each step makes the math visible before adding abstractions.

## Dataset

- **MNIST** — 60,000 train / 10,000 test, 28x28 grayscale, 10 digit classes
- Auto-downloaded via `tensorflow.keras.datasets.mnist` or `torchvision.datasets.MNIST`
- No local image files committed

## Learning roadmap

```mermaid
flowchart LR
    EDA[eda_analysis.ipynb] --> KNN[manual_knn.ipynb]
    KNN --> LR[manual_logistic_reg.ipynb]
    LR --> MLP[manual_mlp.ipynb]
    MLP --> CNN[cnn.ipynb]
    CNN --> CMP[02_mnist_models.ipynb]
    CMP --> VAE[vae.ipynb]
    VAE --> DDPM[ddpm.ipynb]
    DDPM --> GAN[dcgan.ipynb]
    GAN --> LD["vae + 1d unet (latent diff) (1).ipynb"]
```

| Step | Notebook | Track |
|------|----------|-------|
| 1 | `eda_analysis.ipynb` | EDA |
| 2 | `manual_knn.ipynb` | Classification |
| 3 | `manual_logistic_reg.ipynb` | Classification |
| 4 | `manual_mlp.ipynb` | Classification |
| 5 | `cnn.ipynb` | Classification |
| 6 | `02_mnist_models.ipynb` | Comparison |
| 7 | `vae.ipynb` | Generative |
| 8 | `ddpm.ipynb` | Generative |
| 9 | `dcgan.ipynb` | Generative |
| 10 | `vae + 1d unet (latent diff) (1).ipynb` | Generative |

See [`docs/learning_path.md`](docs/learning_path.md) for pacing and prerequisites.

## Classification results

Metrics from saved notebook outputs:

| Model | Metric | Value | Source |
|-------|--------|-------|--------|
| KNN (GPU, k=3) | Test accuracy | **0.9705** | `manual_knn.ipynb` |
| Logistic regression | Test accuracy | **0.9241** | `manual_logistic_reg.ipynb` |
| Manual MLP | Test accuracy | **0.9794** | `manual_mlp.ipynb` |
| SimpleCNN | Test accuracy | **0.9871** | `cnn.ipynb` |
| Comparison CNN | Test accuracy | **0.9913** | `02_mnist_models.ipynb` |

The manual models show a clean learning ladder: KNN does well with almost no training, logistic regression adds an interpretable linear boundary, and the manual MLP adds non-linear structure. The CNN notebooks push accuracy further with spatial filters; the multi-model notebook shows how far classic ML reaches before deep learning takes the lead.

## Generative results

| Model | Metric | Value | Source |
|-------|--------|-------|--------|
| VAE (16-D latent) | Train / val loss | **100.22 / 100.09** | `vae.ipynb` |
| DDPM (T=300) | Avg loss (ep 15) | **0.0454** | `ddpm.ipynb` |
| DCGAN | G loss / D loss | **3.41 / 0.32** | `dcgan.ipynb` |
| Latent diffusion MLP | MSE (ep 150) | **0.224** | `vae + 1d unet (latent diff) (1).ipynb` |

Visual samples (reconstructions, denoising grids, GAN outputs) are exported to [`results/figures/`](results/figures/) and embedded in notebook output cells. See [`docs/results.md`](docs/results.md) for the full metrics table and [`reports/report.pdf`](reports/report.pdf) for the write-up.

## How to run

```bash
cd mnist-generative-models-lab
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run cells top to bottom. **Use the repository root as the working directory** so paths like `./training_results/` resolve correctly.

## Project structure

```text
mnist-generative-models-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/          # 10 Jupyter notebooks (saved outputs)
├── results/
│   ├── RESULTS_INDEX.md
│   ├── figures/        # Collated plots for report and portfolio
│   ├── metrics/        # JSON/CSV metrics from notebook outputs
│   └── tables/
├── scripts/
│   ├── extract_notebook_results.py
│   └── results_manifest.yaml
├── docs/               # Learning path, models, results, workflow
└── reports/
    ├── report.pdf      # Technical report (~2000 words)
    ├── report.md
    ├── report.tex
    ├── references.bib
    └── report_plan.md
```

## Artifact policy

Training outputs are **gitignored** and regenerated locally:

| Path | Created by | Contents |
|------|------------|----------|
| `training_results/` | `vae.ipynb`, latent-diff notebook | `vae_model.pth`, plots, pickles |
| `latent_data/` | Latent-diff notebook | `mnist_latents_*.npy`, label arrays |
| `ddpm_mnist*.pth` | `ddpm.ipynb` | DDPM checkpoints |

Re-run the relevant notebook to recreate artifacts. Metrics in this README come from **saved notebook runs**; full retraining is optional.

## Notebook guide

### `manual_knn.ipynb`
From-scratch KNN in NumPy (`euclidean_distance`, `np.argpartition`) plus a GPU version with `torch.cdist` and `topk`. CPU section uses `N_train=10000`, `N_test=4000`. k-sweep over `[1, 3, 5, 9, 15, 20]` shows the bias-variance tradeoff.

### `manual_logistic_reg.ipynb`
Hand-written softmax regression: `one_hot`, `softmax`, gradient `dZ = (P - Y_batch) / batch_size`, explicit weight updates. Includes misclassified example grid.

### `manual_mlp.ipynb`
Single hidden layer (ReLU + softmax), `H=128`, `lr=0.1`, `epochs=100`. Plots loss, confusion matrix, misclassified digits, and first-layer weight templates.

### `eda_analysis.ipynb`
13-section EDA: shapes, class balance, mean images, pixel std maps, intensity distributions, center of mass, nearest neighbors, PCA variance, atypical samples per class.

### `cnn.ipynb`
PyTorch SimpleCNN: `Conv2d(1,16)` → `Conv2d(16,32)` → `MaxPool` → two linear layers. Five epochs with SGD + momentum. Post-training analysis: ROC, confusion matrix, filter and activation visualizations.

### `02_mnist_models.ipynb`
Benchmarks DummyClassifier, logistic regression, SGD, KNN, PCA+LR, PyTorch MLP, and PyTorch CNN on shared helpers. Results sorted into a DataFrame; top CNN confusion pairs illustrated.

### `vae.ipynb`
16-D variational autoencoder, 50 epochs. Saves to `./training_results/`.

### `ddpm.ipynb`
Pixel-space DDPM with T=300 timesteps and U-Net denoiser, 15 epochs.

### `dcgan.ipynb`
DCGAN generator/discriminator, 50 epochs, latent interpolation.

### `vae + 1d unet (latent diff) (1).ipynb`
VAE plus `LatentDiffusionMLP` in latent space (filename says "1d unet" but the model is an MLP, not a U-Net). Trains VAE, extracts latents, runs 150-epoch diffusion. The opening Colab `drive.mount` cell is guarded and skipped when running locally.

## Parameters that change the story

- **KNN:** `k`, `N_train`, `N_test`
- **Logistic regression:** `learning_rate`, `epochs`, `batch_size`
- **Manual MLP:** `H`, `lr`, `epochs`, `batch_size`
- **CNN:** `epochs`, optimizer `lr` and `momentum`, conv channel sizes
- **Model comparison:** `max_train_samples`, PCA `n_components`, batch size
- **VAE / generative:** latent dimension, epoch count, diffusion timesteps

## Reproducibility

- Results below are from embedded notebook outputs unless you retrain
- MNIST download is deterministic given standard loaders
- TensorFlow and PyTorch both required; see [`docs/dependency_integrity.md`](docs/dependency_integrity.md)
- Package versions: TODO pin after environment check (`requirements.txt`)

## Limitations

- 28x28 grayscale only; not representative of real-world vision
- No FID or Inception Score for generative evaluation
- Latent-diffusion notebook filename is misleading (`LatentDiffusionMLP`, not 1D U-Net)
- VAE training duplicated between `vae.ipynb` and the latent-diff notebook
- `02_mnist_models.ipynb` may contain duplicate CNN rows from re-runs

## Future work

- Deduplicate VAE pipeline (load checkpoint in latent-diff notebook)
- Add FID / IS metrics for generative models
- Fashion-MNIST extension with the same notebook structure
- Pin torch and tensorflow versions; optional `nbconvert` CI smoke test

## Documentation index

| Doc | Purpose |
|-----|---------|
| [`docs/learning_path.md`](docs/learning_path.md) | Ordered syllabus |
| [`docs/models.md`](docs/models.md) | Per-notebook architecture notes |
| [`docs/results.md`](docs/results.md) | Full metrics tables |
| [`docs/branch_pr_plan.md`](docs/branch_pr_plan.md) | 13-branch PR workflow |
| [`docs/smoke_tests.md`](docs/smoke_tests.md) | Lightweight verification |
| [`reports/report_plan.md`](reports/report_plan.md) | ~2000-word report outline |
| [`reports/report.pdf`](reports/report.pdf) | Technical report (PDF) |
| [`reports/report.md`](reports/report.md) | Technical report (Markdown) |
| [`results/RESULTS_INDEX.md`](results/RESULTS_INDEX.md) | Collated figures and metrics from notebook outputs |

## Resume bullet

Built an end-to-end MNIST ML lab from scratch KNN, logistic regression, and MLP through CNN classification (99%+ accuracy) to generative models including VAE, DDPM, DCGAN, and latent diffusion, documented as an interactive Jupyter learning path with a reflective technical report.
