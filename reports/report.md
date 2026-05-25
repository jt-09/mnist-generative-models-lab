# MNIST Generative Models Lab: From Manual Classifiers to VAE, DDPM, and GANs

**Technical report (portfolio project)**  
**Repository:** `mnist-generative-models-lab`  
**Date:** July 2026

---

## Abstract

I built a notebook-driven MNIST lab with two tracks: classification (from-scratch KNN and logistic regression through a PyTorch CNN) and generation (VAE, pixel-space DDPM, DCGAN, and a latent diffusion MLP). On the saved notebook outputs I kept, test accuracy climbs from **0.9705** (GPU KNN, k=3) to **0.9913** (comparison CNN), with logistic regression sitting lower at **0.9241** because it is strictly linear. The generative notebooks train end-to-end and produce readable digit samples, but I did not compute FID or Inception Score; my conclusions about sample quality are qualitative, based on the PNGs exported to `results/figures/`. I document both what worked as a learning path and what I would refactor next (duplicate VAE training, misleading notebook filename, missing quantitative generative metrics).

---

## 1. Introduction

I chose MNIST (LeCun et al., 1998) because it is the standard place to make model behaviour visible before scaling up. This repo is not a production system; it is a curriculum I can walk through in Jupyter: understand the data, implement classical models by hand, add a small CNN, benchmark everything side by side, then explore compression (VAE), denoising (DDPM), adversarial training (DCGAN), and finally diffusion in a 16-D latent space.

![Learning path](../results/figures/00_roadmap/learning_path.png)

*Figure 1: Learning path I followed — EDA → manual classifiers → CNN → comparison → generative models.*

Each notebook preserves embedded outputs so metrics and plots are inspectable without retraining. All figures in this report were copied from those outputs into `results/figures/`; I did not regenerate checkpoints from gitignored `training_results/` or `*.pth` files.

---

## 2. Exploratory data analysis

Before training anything, `eda_analysis.ipynb` loads MNIST and walks through class balance, per-digit galleries, mean templates, pixel variance maps, intensity histograms, and PCA structure. On visual inspection, the class counts are nearly even, which means raw accuracy is a reasonable metric and I do not need heavy rebalancing tricks for this dataset.

| Class balance | Mean digit templates |
|---------------|---------------------|
| ![](../results/figures/01_eda/class_balance.png) | ![](../results/figures/01_eda/mean_digits.png) |

The mean images look like the digits I expect, but PCA plots show substantial overlap between classes in the first two components:

![PCA overlap](../results/figures/01_eda/pca_overlap.png)

That overlap is my main motivation for non-linear models: a single linear boundary in pixel space cannot separate all ten digits cleanly, even though MNIST is "easy" by modern standards.

---

## 3. Classical and manual machine learning

### 3.1 KNN and the bias–variance tradeoff

In `manual_knn.ipynb` I implemented KNN in NumPy and then batched it on GPU with `torch.cdist`. A sweep over k ∈ {1, 3, 5, 9, 15, 20} shows accuracy peaking near k=3 at **0.9705** on the full test set:

![KNN k-sweep](../results/figures/02_knn/k_sweep.png)

Very small k is noisier; very large k oversmooths decision boundaries.

### 3.2 Logistic regression: a deliberate dip in the ladder

I expected accuracy to improve monotonically as models gained capacity. That did not happen. My from-scratch softmax regression in `manual_logistic_reg.ipynb` plateaus at **0.9241**, below KNN and well below the manual MLP. I think that is mostly a capacity issue: logistic regression is a linear classifier in pixel space, and MNIST digits written in different styles still overlap when flattened to 784 dimensions.

![Misclassified digits](../results/figures/03_logistic/misclassified_grid.png)

### 3.3 Manual MLP

Adding one ReLU hidden layer (H=128, 100 epochs) in `manual_mlp.ipynb` raised accuracy to **0.9794**. The first-layer weight templates look like crude stroke detectors:

![MLP diagnostics](../results/figures/04_mlp/weight_templates.png)

### Classification metrics (from `results/metrics/classification.json`)

| Model | Test accuracy | Source notebook |
|-------|---------------|-----------------|
| KNN (GPU, k=3) | 0.9705 | `manual_knn.ipynb` |
| Logistic regression | 0.9241 | `manual_logistic_reg.ipynb` |
| Manual MLP | 0.9794 | `manual_mlp.ipynb` |
| SimpleCNN | 0.9871 | `cnn.ipynb` |
| Comparison CNN | 0.9913 | `02_mnist_models.ipynb` |

---

## 4. CNN baseline

`cnn.ipynb` implements a small SimpleCNN trained five epochs with SGD. Test accuracy reaches **0.9871** by epoch 4. The jump from manual MLP to CNN is modest in absolute terms on MNIST, but I think the spatial inductive bias matters.

![CNN analysis panel](../results/figures/05_cnn/confusion_matrix.png)

![Conv filters](../results/figures/05_cnn/conv_filters.png)

---

## 5. Generative models

After classification, I worked through four generative approaches. I did not run FID or IS; I judged samples by eye on the PNG grids in `results/figures/`.

### 5.1 Variational autoencoder

`vae.ipynb` trains a 16-D VAE for 50 epochs. Final train/val loss: **100.22 / 100.09**. Reconstructions preserve digit identity but look soft.

| Reconstructions | Training curves |
|-----------------|-----------------|
| ![](../results/figures/07_vae/reconstructions.png) | ![](../results/figures/07_vae/training_curves.png) |

### 5.2 Pixel-space DDPM

`ddpm.ipynb`: T=300, 15 epochs, avg loss **0.0454** at epoch 15.

| Loss curve | Sample grid |
|------------|-------------|
| ![](../results/figures/08_ddpm/loss_curve.png) | ![](../results/figures/08_ddpm/denoising_grid.png) |

### 5.3 DCGAN

`dcgan.ipynb`: 50 epochs, G/D loss **3.41 / 0.32**.

| Generated digits | Latent interpolation |
|------------------|---------------------|
| ![](../results/figures/09_dcgan/generated_digits.png) | ![](../results/figures/09_dcgan/latent_interpolation.png) |

### 5.4 Latent diffusion MLP

The notebook `vae + 1d unet (latent diff) (1).ipynb` trains VAE + `LatentDiffusionMLP` (MSE **0.224**). Despite "1d unet" in the filename, the denoiser is an MLP — a naming mistake I should fix.

| VAE reconstructions | Diffusion samples |
|---------------------|-------------------|
| ![](../results/figures/10_latent_diff/vae_reconstructions.png) | ![](../results/figures/10_latent_diff/diffusion_samples.png) |

---

## 6. Comparison and insights

`02_mnist_models.ipynb` benchmarks models on a shared harness. The accuracy ladder and top CNN confusion pairs:

| Accuracy bar chart | Confusion pairs |
|--------------------|-----------------|
| ![](../results/figures/06_comparison/results_bar_chart.png) | ![](../results/figures/06_comparison/cnn_confusion_pairs.png) |

Deep learning wins on MNIST, but classical KNN is already strong. On the generative side: VAEs blur, pixel DDPMs are faithful but heavy, GANs can look sharp but need monitoring, latent diffusion is efficient but inherits VAE quality.

---

## 7. Limitations

- **No FID / IS** — qualitative generative evaluation only.
- **28×28 grayscale** — not representative of real vision tasks.
- **Duplicate VAE training** across `vae.ipynb` and the latent-diff notebook.
- **Misleading latent-diff filename** (MLP, not U-Net).
- **Duplicate CNN rows** possible in the comparison notebook — I cite **0.9913**.
- **Unpinned dependencies** — embedded outputs are the documentation source of truth.

---

## 8. Future work

- Deduplicate VAE pipeline (load checkpoint instead of retraining).
- Add FID or classifier-based generative scores.
- Rename latent-diff notebook; guard Colab cells.
- Fashion-MNIST extension with the same structure.
- Pin torch/tensorflow versions; optional `nbconvert` CI smoke test.

---

## Conclusion

I built an end-to-end MNIST lab that reads like a course project I actually ran. Classification reaches **0.9913** on the comparison CNN; generative models produce inspectable digit samples without me claiming state-of-the-art numbers. Documenting the logistic-regression dip, soft VAE reconstructions, and missing FID metrics is as important as listing peak accuracies — it makes the repo credible for someone reviewing my work.

---

## References

1. LeCun et al. (1998) — Gradient-based learning applied to document recognition.  
2. Kingma & Welling (2014) — Auto-Encoding Variational Bayes.  
3. Ho et al. (2020) — Denoising Diffusion Probabilistic Models.  
4. Goodfellow et al. (2014) — Generative Adversarial Nets.  
5. Radford et al. (2016) — DCGAN.
