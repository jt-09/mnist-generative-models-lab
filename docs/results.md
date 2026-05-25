# Results

All metrics below are taken from **saved notebook cell outputs**. Full retraining is optional; embedded outputs are the source of truth for documentation.

## Classification metrics

| Model | Metric | Value | Source notebook | Notes |
|-------|--------|-------|-----------------|-------|
| KNN (GPU, k=3) | Test accuracy | **0.9705** | `manual_knn.ipynb` | Full test set, batched `torch.cdist` |
| Logistic regression | Test accuracy | **0.9241** | `manual_logistic_reg.ipynb` | Manual softmax |
| Manual MLP | Test accuracy | **0.9794** | `manual_mlp.ipynb` | H=128, 100 epochs |
| SimpleCNN | Test accuracy | **0.9871** | `cnn.ipynb` | Epoch 4 of 5 |
| Comparison CNN | Test accuracy | **0.9913** | `02_mnist_models.ipynb` | Shared benchmark harness |

### Learning ladder (classification)

```
KNN 0.9705 → Logistic 0.9241 → MLP 0.9794 → CNN 0.9871 → Comparison CNN 0.9913
```

Logistic regression sits lower than KNN/MLP on this split because it is a strictly linear boundary; the MLP and CNN add non-linear capacity that MNIST benefits from.

## Generative metrics

| Model | Metric | Value | Source notebook | Notes |
|-------|--------|-------|-----------------|-------|
| VAE | Train / val loss | **100.22 / 100.09** | `vae.ipynb` | 16-D latent, 50 epochs |
| DDPM | Average loss | **0.0454** | `ddpm.ipynb` | T=300, epoch 15 |
| DCGAN | G loss / D loss | **3.41 / 0.32** | `dcgan.ipynb` | 50 epochs |
| Latent diffusion MLP | MSE | **0.224** | `latent_diffusion_mlp.ipynb` | Epoch 150 |

## Qualitative outputs

Visual results live in notebook output cells:

| Content | Notebook |
|---------|----------|
| KNN k-sweep accuracy plot | `manual_knn.ipynb` |
| MLP weight templates, confusion matrix | `manual_mlp.ipynb` |
| CNN filters, activation maps, ROC curves | `cnn.ipynb` |
| Model comparison bar chart / DataFrame | `02_mnist_models.ipynb` |
| VAE reconstructions, training curves | `vae.ipynb` |
| DDPM sample grid | `ddpm.ipynb` |
| DCGAN generated digits, interpolation | `dcgan.ipynb` |
| Latent diffusion samples | `latent_diffusion_mlp.ipynb` |

## Artifact locations (gitignored)

| Path | Created by | Contents |
|------|------------|----------|
| `training_results/` | `vae.ipynb`, latent-diff notebook | `vae_model.pth`, PNGs, pickles |
| `latent_data/` | Latent-diff notebook | `mnist_latents_train/val.npy`, labels |
| `ddpm_mnist*.pth` | `ddpm.ipynb` | DDPM checkpoints |

Regenerate by re-running the corresponding notebook from repo root.

## Known hygiene notes

- `02_mnist_models.ipynb` may list duplicate CNN rows from multiple runs — cite **0.9913** as the comparison CNN result.
- Latent-diffusion notebook filename references "1d unet" but implements `LatentDiffusionMLP`.
- VAE training appears in both `vae.ipynb` and the latent-diffusion notebook; checkpoint reuse avoids redundant training.

## Future metrics (not yet reported)

- FID / Inception Score for generative samples
- Per-class generative accuracy
- Fashion-MNIST transfer results
