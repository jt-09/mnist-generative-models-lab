# Results index

All assets extracted from saved notebook cell outputs (no retraining).

## Report figure checklist

| # | Report figure | Path | Status |
|---|---------------|------|--------|
| 1 | Learning roadmap | `figures/00_roadmap/learning_path.png` | done |
| 2 | KNN k-sweep | `figures/02_knn/k_sweep.png` | done |
| 3 | MLP weight templates | `figures/04_mlp/weight_templates.png` | done |
| 4 | CNN confusion + ROC | `figures/05_cnn/confusion_matrix.png`, `roc_curves.png` | done |
| 5 | Model comparison bar | `figures/06_comparison/results_bar_chart.png` | done |
| 6 | VAE reconstructions | `figures/07_vae/reconstructions.png` | done |
| 7 | DDPM denoising grid | `figures/08_ddpm/denoising_grid.png` | done |
| 8 | DCGAN digits | `figures/09_dcgan/generated_digits.png` | done |
| 9 | Latent diffusion samples | `figures/10_latent_diff/diffusion_samples.png` | done |

## Figures

| File | Source notebook | Cell | Role |
|------|-----------------|------|------|
| `figures/01_eda/class_balance.png` | `notebooks/eda_analysis.ipynb` | 6 | Class balance bar chart (report section 2) |
| `figures/01_eda/mean_digits.png` | `notebooks/eda_analysis.ipynb` | 12 | Mean image per digit |
| `figures/01_eda/pca_overlap.png` | `notebooks/eda_analysis.ipynb` | 27 | PCA variance / 2D projection |
| `figures/02_knn/k_sweep.png` | `notebooks/manual_knn.ipynb` | 2 | GPU k-sweep accuracy plot |
| `figures/03_logistic/misclassified_grid.png` | `notebooks/manual_logistic_reg.ipynb` | 2 | Misclassified examples grid |
| `figures/04_mlp/weight_templates.png` | `notebooks/manual_mlp.ipynb` | 1 | Loss curves, confusion matrix, weight templates |
| `figures/05_cnn/confusion_matrix.png` | `notebooks/cnn.ipynb` | 5 | Post-training analysis panel (confusion, ROC, metrics) |
| `figures/05_cnn/roc_curves.png` | `notebooks/cnn.ipynb` | 5 | Same analysis panel (ROC subplots included) |
| `figures/05_cnn/conv_filters.png` | `notebooks/cnn.ipynb` | 6 | First conv layer filters |
| `figures/06_comparison/cnn_confusion_pairs.png` | `notebooks/02_mnist_models.ipynb` | 24 | Top CNN confusion pairs with examples |
| `figures/07_vae/reconstructions.png` | `notebooks/vae.ipynb` | 4 | VAE reconstruction grid |
| `figures/07_vae/training_curves.png` | `notebooks/vae.ipynb` | 4 | VAE training loss curves |
| `figures/08_ddpm/denoising_grid.png` | `notebooks/ddpm.ipynb` | 18 | DDPM generated samples grid |
| `figures/09_dcgan/generated_digits.png` | `notebooks/dcgan.ipynb` | 6 | Final DCGAN sample grid and loss history |
| `figures/09_dcgan/latent_interpolation.png` | `notebooks/dcgan.ipynb` | 7 | Latent space interpolation |
| `figures/10_latent_diff/vae_reconstructions.png` | `notebooks/latent_diffusion_mlp.ipynb` | 7 | VAE reconstructions in `latent_diffusion_mlp.ipynb` |
| `figures/10_latent_diff/diffusion_samples.png` | `notebooks/latent_diffusion_mlp.ipynb` | 16 | Latent diffusion generated digits |
| `figures/00_roadmap/learning_path.png` | generated | — | Learning path diagram |
| `figures/08_ddpm/loss_curve.png` | ddpm.ipynb streams | — | DDPM loss from epoch logs |
| `figures/06_comparison/results_bar_chart.png` | metrics JSON | — | Accuracy ladder bar chart |

## Metrics summary

### Classification

- `knn_gpu_k3`: **0.9705**
- `logistic_regression`: **0.9241**
- `manual_mlp`: **0.9794**
- `simple_cnn`: **0.9871**
- `comparison_cnn`: **0.9913**

### Generative

- `vae_train_loss`: **100.22**
- `vae_val_loss`: **100.09**
- `ddpm_avg_loss`: **0.0454**
- `dcgan_d_loss`: **0.32**
- `dcgan_g_loss`: **3.41**
- `latent_diff_mse`: **0.225**

## Tables

- `tables/model_comparison.md` — sorted DataFrame from `02_mnist_models.ipynb`

## Exclusions

No files from `training_results/`, `latent_data/`, or `*.pth` checkpoints.