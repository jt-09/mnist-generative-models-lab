# Learning Path

Ordered syllabus for the MNIST Generative Models Lab. Run notebooks top to bottom with the repository root as the working directory.

## Track A — Classification and EDA

| Step | Notebook | Focus | Prerequisites |
|------|----------|-------|---------------|
| 1 | `eda_analysis.ipynb` | Dataset shapes, class balance, PCA, outliers | Basic Python, matplotlib |
| 2 | `manual_knn.ipynb` | NumPy KNN, GPU `torch.cdist` KNN, k-sweep | Linear algebra basics |
| 3 | `manual_logistic_reg.ipynb` | Hand-written softmax regression | Gradients, one-hot encoding |
| 4 | `manual_mlp.ipynb` | Manual 1-hidden-layer MLP (ReLU + softmax) | Backprop from step 3 |
| 5 | `cnn.ipynb` | PyTorch SimpleCNN, filters, activations | PyTorch tensors |
| 6 | `02_mnist_models.ipynb` | Side-by-side benchmark: Dummy, LR, SGD, KNN, PCA+LR, MLP, CNN | sklearn + PyTorch |

**Classification milestone:** CNN reaches **0.9871** test accuracy (`cnn.ipynb`); comparison CNN **0.9913** (`02_mnist_models.ipynb`).

## Track B — Generative Models

| Step | Notebook | Focus | Prerequisites |
|------|----------|-------|---------------|
| 7 | `vae.ipynb` | 16-D VAE, ELBO, reconstructions | PyTorch, Track A helpful |
| 8 | `ddpm.ipynb` | Pixel-space DDPM (T=300, U-Net denoiser) | Diffusion intuition |
| 9 | `dcgan.ipynb` | DCGAN generator/discriminator | Adversarial training basics |
| 10 | `vae + 1d unet (latent diff) (1).ipynb` | VAE + `LatentDiffusionMLP` in latent space | Complete `vae.ipynb` first |

**Generative milestone:** VAE train/val loss **100.22 / 100.09**; DDPM avg loss **0.0454**; DCGAN G/D **3.41 / 0.32**; latent diffusion MSE **0.224**.

## Suggested pacing

- **Week 1:** Steps 1–4 (EDA + manual classifiers)
- **Week 2:** Steps 5–6 (deep learning classification)
- **Week 3:** Steps 7–8 (VAE + DDPM)
- **Week 4:** Steps 9–10 (DCGAN + latent diffusion)

## Working directory

Launch Jupyter from the **repository root** so artifact paths resolve correctly:

```bash
cd mnist-generative-models-lab
jupyter notebook notebooks/
```

Paths like `./training_results/vae_model.pth` and `./latent_data/` are relative to the repo root.

## Optional extensions

- Fashion-MNIST swap-in (same pipeline shape)
- FID / IS metrics for generative outputs
- Deduplicate VAE training between `vae.ipynb` and the latent-diffusion notebook by loading a saved checkpoint
