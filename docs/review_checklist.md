# Review Checklist

Use before merging each PR or before publishing `main`.

## Repository structure

- [ ] `notebooks/` contains all notebooks with **original filenames** (not renamed)
- [ ] `README.md` documents both classification and generative tracks
- [ ] `requirements.txt` at repo root (not parent monorepo path)
- [ ] `.gitignore` excludes `training_results/`, `latent_data/`, `*.pth`, `*.pkl`, `*.npy`
- [ ] `docs/` and `reports/` present

## Notebook quality

- [ ] Embedded outputs preserved (metrics visible without retraining)
- [ ] Cells run top-to-bottom when cwd = repo root
- [ ] No committed secrets or API keys
- [ ] Colab `drive.mount` guarded or removed in latent-diff notebook
- [ ] No large binary artifacts inside `.ipynb` files (check file size)

## Metrics accuracy (from outputs only)

| Metric | Expected | Notebook |
|--------|----------|----------|
| KNN accuracy | 0.9705 | `manual_knn.ipynb` |
| Logistic accuracy | 0.9241 | `manual_logistic_reg.ipynb` |
| MLP accuracy | 0.9794 | `manual_mlp.ipynb` |
| CNN accuracy | 0.9871 | `cnn.ipynb` |
| Comparison CNN | 0.9913 | `02_mnist_models.ipynb` |
| VAE train/val loss | 100.22 / 100.09 | `vae.ipynb` |
| DDPM avg loss | 0.0454 | `ddpm.ipynb` |
| DCGAN G/D loss | 3.41 / 0.32 | `dcgan.ipynb` |
| Latent diff MSE | 0.224 | `vae + 1d unet (latent diff) (1).ipynb` |

- [ ] `docs/results.md` matches table above
- [ ] `README.md` results tables match table above
- [ ] No invented metrics

## Documentation

- [ ] `docs/learning_path.md` lists notebooks in pedagogical order
- [ ] `docs/models.md` has one section per notebook
- [ ] `docs/branch_pr_plan.md` lists 13 branches
- [ ] `docs/dependency_integrity.md` documents cwd and artifact paths
- [ ] `reports/report_plan.md` outline complete

## Dependencies

- [ ] `pip install -r requirements.txt` succeeds (or documented blockers)
- [ ] Import smoke test passes (see `docs/smoke_tests.md`)
- [ ] TensorFlow + PyTorch coexistence noted in README limitations

## Git hygiene

- [ ] No `training_results/` or `latent_data/` tracked
- [ ] No `*.pth` or large `.npy` committed
- [ ] `.ipynb_checkpoints/` gitignored
- [ ] `venv/` not tracked

## Portfolio readiness

- [ ] GitHub description set
- [ ] Repo is public (if intended for portfolio)
- [ ] Resume bullet in README is accurate
- [ ] Limitations section mentions: no FID, latent-diff filename, VAE duplication
