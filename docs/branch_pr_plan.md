# Branch and PR Plan

Thirteen squash-merged branches document project evolution. `main` is the polished final state after all PRs merge.

## Merge order

```
setup → eda → knn → logistic → mlp → cnn → comparison → vae → ddpm → dcgan → latent-diff → docs → report
```

## Branch table

| # | Branch | PR title | Scope | Notebook(s) | Review focus | Smoke check |
|---|--------|----------|-------|-------------|--------------|-------------|
| 1 | `setup/repo-structure` | Initialize MNIST lab repo | `.gitignore`, `requirements.txt`, `reports/`, `notebooks/` layout | — | Directory tree matches plan | Parse repo structure |
| 2 | `analysis/eda` | EDA notebook | `eda_analysis.ipynb` | `eda_analysis.ipynb` | Outputs preserved; MNIST loads | JSON parse notebook |
| 3 | `model/manual-knn` | Manual KNN | `manual_knn.ipynb` | `manual_knn.ipynb` | Metric 0.9705 in outputs | Parse notebook |
| 4 | `model/manual-logistic-regression` | Manual logistic regression | `manual_logistic_reg.ipynb` | `manual_logistic_reg.ipynb` | Metric 0.9241 cited | Parse notebook |
| 5 | `model/manual-mlp` | Manual MLP | `manual_mlp.ipynb` | `manual_mlp.ipynb` | Metric 0.9794 cited | Parse notebook |
| 6 | `model/cnn-baseline` | PyTorch CNN | `cnn.ipynb` | `cnn.ipynb` | Metric 0.9871 cited | Parse notebook |
| 7 | `model/model-comparison` | Multi-model comparison | `02_mnist_models.ipynb` | `02_mnist_models.ipynb` | CNN 0.9913; note duplicate rows | Parse notebook |
| 8 | `model/vae` | Variational autoencoder | `vae.ipynb` | `vae.ipynb` | Loss 100.22/100.09; artifact paths | Parse notebook |
| 9 | `model/ddpm` | Denoising diffusion | `ddpm.ipynb` | `ddpm.ipynb` | Loss 0.0454; checkpoints gitignored | Parse notebook |
| 10 | `model/dcgan` | DCGAN | `dcgan.ipynb` | `dcgan.ipynb` | G/D 3.41/0.32 | Parse notebook |
| 11 | `model/latent-diffusion` | Latent diffusion MLP | `latent_diffusion_mlp.ipynb` | latent-diff notebook | MSE 0.224; Colab cell guarded | Parse notebook |
| 12 | `docs/learning-roadmap` | Learning roadmap and README | `README.md`, `docs/*` | — | Both tracks documented; metrics cited | README links valid |
| 13 | `docs/report-plan` | Report plan | `reports/report_plan.md` | — | Outline complete (~2000 words) | File exists |

## PR description template

```markdown
## Summary
- <one-line scope>

## Notebook / files
- `notebooks/<filename>`

## Metrics verified (from outputs)
- <metric>: <value>

## Checklist
- [ ] Notebook outputs preserved
- [ ] No committed checkpoints (.pth, .npy in training_results/)
- [ ] Colab-only cells guarded or removed (latent-diff only)
- [ ] docs/results.md metrics match outputs
```

## Alternative workflow

All changes may land directly on `main` if a simpler workflow is preferred. The branch plan above is for portfolio narrative and review granularity.

## Post-merge `main` checklist

- [ ] All 10 notebooks in `notebooks/` with original filenames
- [ ] `README.md` covers classification + generative tracks
- [ ] `requirements.txt` lists torch + tensorflow
- [ ] `training_results/` and `latent_data/` gitignored
- [ ] `docs/learning_path.md` complete
- [ ] `reports/report_plan.md` present
- [ ] GitHub description: "MNIST lab: manual ML to VAE, DDPM, DCGAN, latent diffusion"
