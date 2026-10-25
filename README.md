# FashionMNIST Clustering – Unsupervised Learning Project
Comparing dimensionality reduction (SVD and an autoencoder) with clustering (K-Means, GMM and DBSCAN) on 7 FashionMNIST clothing classes, to find the pipeline that best recovers the true categories.

We built and compared complete unsupervised pipelines on a 7-class subset of [FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html). Each pipeline first reduces the 784-pixel images to a low-dimensional representation and then clusters them. We evaluated each pipeline against the true labels to find the best combination.

**Classes used:** Trouser, Pullover, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Pipeline

**1. Dimensionality reduction**
- **Truncated SVD:** a linear baseline.
- **Autoencoder (PyTorch):** a neural network that learns a non-linear latent representation. We compared a narrow bottleneck (d = 2) with a wider one (d = 32).

**2. Clustering**
- **K-Means**
- **Gaussian Mixture Model (GMM)**
- **DBSCAN**

Two reduction methods times three clustering algorithms gives **6 pipelines** in total.

**3. Evaluation**
- **Clustering accuracy:** predicted clusters are matched to true labels with the Hungarian algorithm (`linear_sum_assignment`). The matching function is checked with unit-test asserts.
- **ARI** (Adjusted Rand Index), **NMI** (Normalized Mutual Information), and **Silhouette score**.
- Visual checks: 2D plots of the embeddings, autoencoder reconstructions, and linear interpolation in the latent space.

---

## Key Results

| Finding | Details |
|---|---|
| Best pipeline | **Autoencoder + GMM**: Accuracy 0.696, ARI 0.515, NMI 0.624 |
| Autoencoder vs. SVD | The autoencoder's non-linear features consistently gave better clustering than SVD with GMM and DBSCAN. |
| DBSCAN | Weakest method (accuracy 0.37–0.41). FashionMNIST doesn't have well-separated dense regions, which density-based clustering needs. |
| Latent size | d = 32 gave much sharper reconstructions, semantic interpolations between garments, and tighter class clusters than d = 2. |

**Conclusion:** for complex image data, an autoencoder with a reasonably wide latent space (d = 32) combined with a flexible probabilistic clusterer (GMM) worked best. It balanced compression with preserving the visual meaning of the images.

---

## Tools & Libraries

PyTorch, torchvision, scikit-learn, SciPy, NumPy, pandas, Matplotlib

## How to Run

The notebook was written for **Google Colab** and is saved with all outputs, so you can read the results directly on GitHub.

To run it yourself, open it in [Google Colab](https://colab.research.google.com/) and run all cells. FashionMNIST downloads automatically through `torchvision`. A GPU speeds up autoencoder training but isn't required.

All random seeds are fixed at 42 for reproducibility.
