# Technical Report Plan

Target: ~2000 words. Audience: ML recruiter, technical interviewer, portfolio reviewer.  
Deliverables: this outline now; `report.tex` and `report.pdf` later.

## Section outline

| # | Section | Words | Content | Figures / sources |
|---|---------|-------|---------|-------------------|
| 1 | Introduction | 200 | Why MNIST for pedagogy; two-track lab scope (classification → generative) | — |
| 2 | EDA findings | 200 | Class balance, PCA overlap, atypical digits | `eda_analysis.ipynb` plots |
| 3 | Classical and manual ML | 300 | KNN → logistic → MLP ladder; bias-variance intuition | k-sweep plot (`manual_knn.ipynb`); MLP weight templates |
| 4 | CNN baseline | 200 | SimpleCNN architecture; 98%+ accuracy; filter visualizations | Confusion matrix, activation maps (`cnn.ipynb`) |
| 5 | Generative models | 500 | VAE (16-D ELBO), pixel DDPM (T=300), DCGAN, latent diffusion MLP | VAE reconstructions; DDPM samples; DCGAN grid; latent samples |
| 6 | Comparison and insights | 250 | When deep learning wins; comparison table (0.9913 CNN); generative tradeoffs | `02_mnist_models.ipynb` DataFrame |
| 7 | Limitations | 200 | No FID; 28×28 grayscale only; VAE duplication; misleading latent-diff filename | — |
| 8 | Future work | 150 | Deduped VAE pipeline; FID metrics; Fashion-MNIST extension | — |

**Total:** ~2000 words

## Results to cite (from notebook outputs)

### Classification

| Model | Metric | Source |
|-------|--------|--------|
| KNN (k=3, GPU) | 0.9705 | `manual_knn.ipynb` |
| Logistic regression | 0.9241 | `manual_logistic_reg.ipynb` |
| Manual MLP | 0.9794 | `manual_mlp.ipynb` |
| SimpleCNN | 0.9871 | `cnn.ipynb` |
| Comparison CNN | 0.9913 | `02_mnist_models.ipynb` |

### Generative

| Model | Metric | Source |
|-------|--------|--------|
| VAE | train/val loss 100.22 / 100.09 | `vae.ipynb` |
| DDPM | avg loss 0.0454 (ep 15) | `ddpm.ipynb` |
| DCGAN | G/D loss 3.41 / 0.32 | `dcgan.ipynb` |
| Latent diffusion MLP | MSE 0.224 (ep 150) | `vae + 1d unet (latent diff) (1).ipynb` |

## Figure list

1. **Learning roadmap diagram** — EDA → manual ML → CNN → generative (mermaid or exported PNG)
2. **KNN k-sweep** — `manual_knn.ipynb` output
3. **MLP first-layer filters** — `manual_mlp.ipynb` output
4. **CNN confusion matrix + ROC** — `cnn.ipynb` output
5. **Model comparison bar chart** — `02_mnist_models.ipynb` output
6. **VAE reconstruction grid** — `vae.ipynb` output
7. **DDPM sample progression** — `ddpm.ipynb` output
8. **DCGAN generated digits** — `dcgan.ipynb` output
9. **Latent diffusion samples** — latent-diff notebook output

## Source files

- All 10 notebooks in `notebooks/`
- `docs/results.md` — metrics tables
- `docs/models.md` — architecture notes
- `docs/learning_path.md` — syllabus ordering
- Monorepo `PROJECT_SUMMARY.md` §5 (MNIST section) — historical context

## TODOs before writing `report.tex`

- [x] Export key figures from notebook outputs to `results/figures/` (see `results/RESULTS_INDEX.md`)
- [x] Verify all metrics still match embedded outputs after any notebook edit
- [x] Decide whether to include GPU vs CPU KNN comparison in §3 (GPU k-sweep used in report)
- [x] Add bibliography entries for VAE, DDPM, DCGAN original papers (`reports/references.bib`)
- [ ] Pin dependency versions in `requirements.txt` and note in §7 reproducibility (noted as limitation in report)

## Suggested narrative arc

1. Start with EDA showing MNIST is nearly balanced but PCA clusters overlap → motivates non-linear models.
2. Walk the manual ladder with explicit equations (softmax gradient, ReLU MLP backprop).
3. Show CNN jumps accuracy to 0.9871+ with spatial inductive bias.
4. Pivot to generation: compression (VAE) → denoising (DDPM) → adversarial (DCGAN) → efficient latent diffusion.
5. Close with honest limitations and a concrete future-work roadmap.

## Resume bullet (for report abstract)

Built an end-to-end MNIST ML lab—from scratch KNN, logistic regression, and MLP through CNN classification (99%+ accuracy) to generative models including VAE, DDPM, DCGAN, and latent diffusion—documented as an interactive Jupyter learning path.
