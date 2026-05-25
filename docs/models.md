# Models Reference

One subsection per notebook. Filenames are **original** (not renamed).

---

## `eda_analysis.ipynb`

**Track:** EDA  
**Framework:** TensorFlow/Keras MNIST loader, NumPy, matplotlib, sklearn PCA

Exploratory analysis across 13 sections: shapes and pixel ranges, class balance, sample grids, mean images per digit, per-pixel standard deviation, intensity histograms, center-of-mass scatter, nearest neighbors in pixel space, PCA explained variance, PC1 vs PC2 overlap, and atypical samples per class.

**Key outputs:** Class balance plots, PCA structure, unusual-digit montages.

---

## `manual_knn.ipynb`

**Track:** Classification  
**Framework:** NumPy (CPU), PyTorch (GPU batched KNN)

From-scratch KNN with `euclidean_distance` and `np.argpartition` for top-k neighbors. CPU run uses `N_train=10000`, `N_test=4000`. GPU section uses `torch.cdist` and `topk` with `knn_predict_batched`. k-sweep over `[1, 3, 5, 9, 15, 20]`.

**Reported metric:** Test accuracy **0.9705** (k=3, GPU full test set).

**Tunable:** `k`, `N_train`, `N_test`, batch size.

---

## `manual_logistic_reg.ipynb`

**Track:** Classification  
**Framework:** NumPy (manual softmax regression)

Implements `one_hot`, `softmax`, and explicit gradient updates `dZ = (P - Y_batch) / batch_size`, `W -= lr * dW`. Includes `show_misclassified_examples` grid.

**Reported metric:** Test accuracy **0.9241**.

**Tunable:** `learning_rate`, `epochs`, `batch_size`.

---

## `manual_mlp.ipynb`

**Track:** Classification  
**Framework:** NumPy

Single hidden layer: ReLU + softmax, `H=128`, `lr=0.1`, `epochs=100`, `batch_size=256`. Visualizes loss/accuracy curves, confusion matrix, misclassified montage, and first-layer weight templates (`W1[:, i].reshape(28, 28)`).

**Reported metric:** Test accuracy **0.9794**.

**Tunable:** `H`, `lr`, `epochs`, `batch_size`.

---

## `cnn.ipynb`

**Track:** Classification  
**Framework:** PyTorch

`SimpleCNN`: `Conv2d(1,16)` → `Conv2d(16,32)` → `MaxPool` → `Linear(32*14*14, 128)` → `Linear(128, 10)`. Trained 5 epochs with SGD + momentum. Post-training: per-class accuracy, classification report, confusion matrix, ROC (one-vs-rest), confidence histograms, filter/activation hooks.

**Reported metric:** Test accuracy **0.9871** (epoch 4).

**Tunable:** `epochs`, optimizer `lr`/`momentum`, channel widths.

---

## `02_mnist_models.ipynb`

**Track:** Classification / comparison  
**Framework:** sklearn + PyTorch

Benchmarks DummyClassifier, logistic regression, SGDClassifier, KNN, PCA + logistic regression, PyTorch MLP, and PyTorch CNN on a shared evaluation harness. Results collected into a sorted DataFrame. Final section analyzes top CNN confusion pairs with example images.

**Reported metric:** CNN test accuracy **0.9913**.

**Note:** Notebook may contain duplicate CNN rows from re-runs — use the highest consistent entry when comparing.

**Tunable:** `max_train_samples`, PCA `n_components`, DataLoader batch size.

---

## `vae.ipynb`

**Track:** Generative  
**Framework:** PyTorch, torchvision MNIST

16-dimensional latent VAE trained 50 epochs. Saves checkpoint and plots to `./training_results/` (`vae_model.pth`, PNGs, pickles).

**Reported metrics:** Train loss **100.22**, validation loss **100.09**.

**Artifacts:** `./training_results/vae_model.pth`, `./training_results/*.png`, `./training_results/*.pkl`

---

## `ddpm.ipynb`

**Track:** Generative  
**Framework:** PyTorch, torchvision MNIST

Pixel-space denoising diffusion probabilistic model: T=300 timesteps, U-Net denoiser, 15 training epochs. May save `ddpm_mnist.pth` / `ddpm_mnist_model.pth` (gitignored).

**Reported metric:** Average loss **0.0454** (epoch 15).

---

## `dcgan.ipynb`

**Track:** Generative  
**Framework:** TensorFlow/Keras MNIST

DCGAN with separate generator and discriminator, 50 epochs. Includes latent interpolation visualizations.

**Reported metrics:** Generator loss **3.41**, discriminator loss **0.32** (final epoch).

---

## `latent_diffusion_mlp.ipynb`

**Track:** Generative  
**Framework:** PyTorch

Pipeline: train or load VAE → extract latents to `./latent_data/` → train class-conditioned latent diffusion MLP (150 epochs). May contain a Colab `drive.mount` cell — guard or remove for local runs.

**Reported metric:** Latent diffusion MSE **0.224** (epoch 150).

**Artifacts:** `./training_results/vae_model.pth`, `./latent_data/mnist_latents_*.npy`

**Overlap:** VAE section duplicates `vae.ipynb`; prefer loading checkpoint after first VAE run.
