# Dependency Integrity

How Python packages, data loaders, and artifact paths interact across notebooks.

## Python environment

| Item | Status |
|------|--------|
| Python version | TODO: verify — TensorFlow and PyTorch coexist in notebooks |
| Virtual env | Recommended: `python -m venv venv` at repo root |
| Install | `pip install -r requirements.txt` |

## Required packages

From `requirements.txt`:

| Package | Used by |
|---------|---------|
| `numpy` | All manual notebooks, EDA |
| `pandas` | EDA, comparison notebook |
| `seaborn` | EDA, plotting |
| `matplotlib` | All notebooks |
| `scikit-learn` | `02_mnist_models.ipynb`, EDA (PCA) |
| `jupyter` | Notebook execution |
| `torch` | GPU KNN, CNN, VAE, DDPM, latent diffusion |
| `torchvision` | PyTorch MNIST loaders |
| `tensorflow` | Keras MNIST in several classification + DCGAN notebooks |
| `tqdm` | Training loops in generative notebooks |

**Version pins:** All packages marked `TODO: pin after environment check` in `requirements.txt`. Do not invent versions until a clean install is verified on target hardware.

## Data loading

No local image files are committed. MNIST auto-downloads:

| Loader | Notebooks |
|--------|-----------|
| `tensorflow.keras.datasets.mnist` | `manual_knn.ipynb` (CPU), `manual_logistic_reg.ipynb`, `eda_analysis.ipynb`, `cnn.ipynb`, `dcgan.ipynb`, `02_mnist_models.ipynb` |
| `torchvision.datasets.MNIST` | `manual_knn.ipynb` (GPU), `vae.ipynb`, `ddpm.ipynb`, latent-diff notebook |

First run requires network access. Caches live in user home (`~/.keras`, `~/.cache/torch`).

## Working directory contract

**Expected cwd: repository root.**

| Path | Notebooks | Gitignored |
|------|-----------|------------|
| `./training_results/vae_model.pth` | `vae.ipynb`, latent-diff | Yes |
| `./training_results/*.png`, `*.pkl` | `vae.ipynb` | Yes |
| `./latent_data/mnist_latents_*.npy` | Latent-diff | Yes |
| `ddpm_mnist.pth`, `ddpm_mnist_model.pth` | `ddpm.ipynb` | Yes |

If Jupyter is started inside `notebooks/`, paths break unless updated to `../training_results/`. **Prefer:** launch from repo root:

```bash
jupyter notebook notebooks/
```

## Framework coexistence

TensorFlow and PyTorch are both required. They do not share tensors — each notebook sticks to one framework per training loop. No cross-framework checkpoint loading.

## Colab-specific code

`vae + 1d unet (latent diff) (1).ipynb` may import `google.colab.drive`. For local runs:

```python
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
```

Or remove the mount cell entirely for Desktop use.

## Notebook cross-references

Notebooks do **not** import each other by filename. The only coupling is:

1. Latent-diff notebook expects VAE weights at `./training_results/vae_model.pth` (from `vae.ipynb` or its own VAE section).
2. Latent-diff expects `./latent_data/*.npy` after extraction cells run.

## Integrity checks before release

```bash
# Parse all notebooks
python -c "
import json, pathlib
for p in pathlib.Path('notebooks').glob('*.ipynb'):
    json.load(open(p, encoding='utf-8'))
    print('OK', p.name)
"

# Import smoke test
python -c "import numpy, pandas, torch, torchvision, tensorflow, sklearn, matplotlib, seaborn, tqdm"
```

## Known gaps

- [ ] Pin torch and tensorflow after environment check
- [ ] Guard or remove Colab mount in latent-diff notebook
- [ ] Optional: dedupe VAE training between `vae.ipynb` and latent-diff notebook
- [ ] Optional: rename latent-diff notebook to accurate filename (breaking change for links)
