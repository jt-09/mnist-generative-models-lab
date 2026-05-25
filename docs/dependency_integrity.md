# Dependency Integrity

How Python packages, data loaders, and artifact paths interact across notebooks.

## Python environment

| Item | Status |
|------|--------|
| Python version | **3.13.5** tested (TensorFlow and PyTorch coexist in notebooks) |
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

**Version pins:** Locked in `requirements.txt` (tested on Python 3.13.5, Windows).

## Data loading

No local image files are committed. MNIST auto-downloads:

| Loader | Notebooks |
|--------|-----------|
| `tensorflow.keras.datasets.mnist` | `manual_knn.ipynb` (CPU), `manual_logistic_reg.ipynb`, `eda_analysis.ipynb`, `cnn.ipynb`, `dcgan.ipynb`, `02_mnist_models.ipynb` |
| `torchvision.datasets.MNIST` | `manual_knn.ipynb` (GPU), `vae.ipynb`, `ddpm.ipynb`, `latent_diffusion_mlp.ipynb` |

First run requires network access. Caches live in user home (`~/.keras`, `~/.cache/torch`).

## Working directory contract

**Expected cwd: repository root.**

| Path | Notebooks | Gitignored |
|------|-----------|------------|
| `./training_results/vae_model.pth` | `vae.ipynb`, `latent_diffusion_mlp.ipynb` | Yes |
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

`latent_diffusion_mlp.ipynb` may import `google.colab.drive`. For local runs:

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

- [x] Pin torch and tensorflow in `requirements.txt` (Python 3.13.5)
- [ ] Guard or remove Colab mount in `latent_diffusion_mlp.ipynb`
- [ ] Optional: dedupe VAE training between `vae.ipynb` and `latent_diffusion_mlp.ipynb`
