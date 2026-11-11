# AI5707 — Lab 8: Classification and PCA

**Pattern Recognition and Machine Learning | Assignment 8**  
**Due:** 10 November 2026

This README documents the **actual implementation** in `src/` and the four Jupyter notebooks. It explains each algorithm's purpose, mathematical formulas, the corresponding code, and how it is used in the experiments. The mathematical framing of GMM/EM, K-means, Fisher LDA and SVM follows the supplied professor's handwritten notes; implementation-specific differences are marked explicitly.

## 1. Objective

Compare **Gaussian Mixture Model (GMM), Linear Discriminant Analysis (LDA), K-Nearest Neighbours (KNN), and Support Vector Machine (SVM)** on four datasets. Repeat each classifier experiment after projecting features onto the top **K principal components (PCA)**. Compare validation/test accuracies and confusion matrices, and explain the observed differences.

**Libraries:** NumPy for calculations, Matplotlib for plotting, Pillow for reading PNG images, and Python's standard library for files. **No SciPy, pandas, or scikit-learn.** The algorithms are implemented as reusable Python classes under `src/`.

## 2. Folder layout

```text
lab8/
├── data/                      # Existing extracted datasets; not included in code-only package
│   ├── synthetic/data.pkl
│   ├── mnist/TEAM_3/train/    # Digit subfolders: 1, 8, 9
│   ├── image_features/       # train/val/test data and label CSVs
│   └── tidigits/             # 3.npz, 7.npz, 9.npz
├── src/
│   ├── gmm.py                # GMM, ClassConditionalGMM, FrameGMM
│   ├── kmeans.py             # KMeans
│   ├── knn.py                # KNN
│   ├── lda.py                # Gaussian shared-covariance LDA classifier
│   ├── svm.py                # BinarySVM and multiclass SVM
│   ├── pca.py                # StandardScaler and PCA
│   ├── speech.py             # SpeechSummary and FrameTransformer
│   ├── lcg.py                # Linear Congruential Generator (LCG)
│   └── evaluation.py         # accuracy and confusion
├── 01_Synthetic.ipynb
├── 02_MNIST.ipynb
├── 03_Image_Features.ipynb
└── 04_TIDIGITS.ipynb
```

Run notebooks **from the `lab8/` root** so `from src.gmm import GMM` works. The notebooks assume the displayed `data/` paths. If your extracted folder names differ, change the corresponding data-loading cell; no environment/path setup is needed.

## 3. Datasets and representations

| Dataset | Samples | Classes | Original representation | PCA dimension used in notebook |
|---|---:|---|---|---:|
| Synthetic | 20,000 | 0, 1 | 16-dimensional vectors | 8 |
| MNIST | 21,660 | 1, 8, 9 | 28×28 grayscale image → 784 pixels | 40 |
| Image features | 3,000 | 0–4 | 36-dimensional supplied features | 12 |
| TIDIGITS | 1,350 | 3, 7, 9 | Variable-length sequences of 39-dimensional frames | 20 for 78-D summaries; 15 for 39-D frames |

Synthetic data and TIDIGITS are split into training/validation/test in the notebooks. MNIST and image features use the supplied partitions. TIDIGITS uses **different representations**: 78-D recording summaries for LDA/KNN/SVM, and ordered 39-D frame sequences for GMM. Therefore the speech comparison is partly a **representation comparison**, not just a classifier comparison.

## 4. Shared preprocessing — `src/pca.py`

### 4.1 Standardization (`StandardScaler`)

For feature `j`, estimate its training mean and standard deviation:

\[
\mu_j=\frac1N\sum_{i=1}^N x_{ij},\qquad
\sigma_j=\sqrt{\frac1N\sum_{i=1}^N(x_{ij}-\mu_j)^2}.
\]

Transform each feature using:

\[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
\]

**Why:** KNN uses distances; SVM uses dot products/kernels; PCA is driven by variance. Large-scale features can dominate these operations. The implementation replaces near-zero standard deviations with 1 to avoid division by zero.

**Code:** `StandardScaler().fit(X_train)` learns `mean_` and `scale_`; `.transform(...)` applies the **same training statistics** to validation/test. Never fit a scaler separately on test data.

### 4.2 PCA (`PCA`)

PCA finds orthogonal directions of maximum variance and projects the data into a smaller space. It is **unsupervised**: class labels are not used.

1. Center training observations: \(\widetilde X=X-\mathbf 1\mu^T\).
2. Compute covariance: \(S=\frac{1}{N-1}\widetilde X^T\widetilde X\).
3. Solve the eigenvalue equation: \(S v_j=\lambda_j v_j\).
4. Sort eigenvalues descending: \(\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\).
5. Form \(W_K=[v_1,\ldots,v_K]\), then project: \(Z=(X-\mu)W_K\).

Retained explained-variance ratio:

\[
R_K=\frac{\sum_{j=1}^K\lambda_j}{\sum_{j=1}^d\lambda_j}.
\]

**Why:** Compare classification on all original features versus a compact representation. PCA may reduce noise and computation, but may also remove low-variance directions that distinguish classes.

**Code:** `np.linalg.eigh` in `PCA.fit`, `components_` for top-K eigenvectors, `explained_variance_ratio_` for retained variance, and `transform` for projection. PCA is fitted on training features only. The PCA branch follows the same scaling step as the original branch.

## 5. K-means — `src/kmeans.py`

**Purpose:** Unsupervised grouping of vectors into K clusters. It is **not one of the four required standalone classifiers** in the notebooks; the module is available for clustering/centroid analysis and for understanding mixture-model initialization.

**Hard assignment** of observation \(x_n\) to its closest centroid:

\[
r_{nk}=\begin{cases}1,&k=\arg\min_j\|x_n-\mu_j\|^2\\0,&\text{otherwise.}\end{cases}
\]

**Distortion (objective):**

\[
J=\sum_{n=1}^N\sum_{k=1}^K r_{nk}\|x_n-\mu_k\|^2.
\]

**Centroid update:**

\[
\mu_k=\frac{\sum_n r_{nk}x_n}{\sum_n r_{nk}}.
\]

**Algorithm:** randomly select K initial observations as centroids → assign each observation to nearest centroid → recompute centroids → stop when the largest centroid movement is below `tol` or `max_iter` is reached.

**Code:** `KMeans(K=3).fit(X)` stores `centers` and `n_iter_`; `predict(X)` returns cluster IDs. Distance is computed efficiently via \(\|x-c\|^2=\|x\|^2+\|c\|^2-2x^Tc\). Empty clusters retain their previous centroids. **Important:** the current GMM code initializes means by random observations, **not** by calling this KMeans class.

## 6. Gaussian Mixture Model — `src/gmm.py`

**Purpose:** Represent a potentially multimodal class density using a weighted sum of Gaussian components, then classify by comparing class likelihoods. This follows the mixture model and **Expectation–Maximization (EM)** updates in the professor's GMM notes (particularly pages 7–12).

### 6.1 Density model

\[
p(x\mid\omega_c)=\sum_{k=1}^K\pi_{ck}\,\mathcal N(x\mid\mu_{ck},\Sigma_{ck}),
\qquad \sum_k\pi_{ck}=1.
\]

Here \(\pi_{ck}\) is a scalar mixture weight, \(\mu_{ck}\) is a d-dimensional mean vector, and \(\Sigma_{ck}\) is a covariance matrix. Each class has its **own** mixture model.

Multivariate Gaussian:

\[
\mathcal N(x\mid\mu,\Sigma)=\frac{\exp[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)]}{(2\pi)^{d/2}|\Sigma|^{1/2}}.
\]

### 6.2 E-step: responsibilities

\[
\gamma_{nk}=P(z_n=k\mid x_n)
=\frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}{\sum_{j=1}^K\pi_j\mathcal N(x_n\mid\mu_j,\Sigma_j)}.
\]

`gamma[n,k]` is the probability that component k explains observation n. Unlike K-means' hard assignment, GMM uses **soft assignments**, with \(\sum_k\gamma_{nk}=1\).

### 6.3 M-step: parameter updates

\[
N_k=\sum_n\gamma_{nk},\qquad
\pi_k=\frac{N_k}{N},\qquad
\mu_k=\frac1{N_k}\sum_n\gamma_{nk}x_n.
\]

Full-covariance update **in the professor's notes**:

\[
\Sigma_k=\frac1{N_k}\sum_n\gamma_{nk}(x_n-\mu_k)(x_n-\mu_k)^T.
\]

**Implementation difference:** `gmm.py` currently uses **diagonal covariance**, updating feature variances as

\[
\sigma_{kj}^2=\max\!\left(\frac{\sum_n\gamma_{nk}x_{nj}^2}{N_k}-\mu_{kj}^2,\ \texttt{reg}\right).
\]

This is cheaper than full covariance and avoids estimating feature-to-feature covariance, but cannot directly model tilted Gaussian components. It is a restricted version of the professor's full-covariance formula.

### 6.4 Log-likelihood and prediction

\[
\ell(\theta)=\sum_{n=1}^N\log\!\left[\sum_k\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)\right].
\]

The code uses **log-sum-exp** for numerical stability:

\[
\operatorname{LSE}(a_1,\ldots,a_K)=m+\log\sum_k e^{a_k-m},\quad m=\max_k a_k.
\]

Classification uses Bayes' rule in log space:

\[
\widehat c=\arg\max_c\big[\log p(x\mid\omega_c)+\log P(\omega_c)\big].
\]

**Code:** `GMM(K=3, reg=1e-3, max_iter=100, tol=1e-4).fit(X_class)` learns `weights_`, `means_`, `vars_`, and `log_likelihoods`. `score_samples(X)` returns **per-sample log density**. The notebooks train one GMM per class and add log class priors. The fit loop stops when the change in mean log-likelihood falls below `tol` or reaches `max_iter`.

**Speech use:** A GMM is fitted to pooled **training frames** from each digit. At prediction, the code averages per-frame log densities for a recording and adds a log class prior. This assumes frame likelihoods are informative and does not explicitly model temporal ordering. The `FrameGMM` class is available, but the notebook uses its own class-wise fitting/scoring cells.

## 7. Linear Discriminant Analysis — `src/lda.py`

**Purpose:** Classify using linear decision boundaries under a Gaussian class model with a **shared covariance matrix**. It works well when these assumptions approximately hold.

### 7.1 What the professor's Fisher LDA notes derive

Within-class scatter for class c:

\[
S_c=\sum_{x\in D_c}(x-m_c)(x-m_c)^T,\qquad S_W=\sum_c S_c.
\]

For two classes, between-class scatter:

\[
S_B=(m_1-m_2)(m_1-m_2)^T.
\]

Fisher's separation criterion:

\[
J(w)=\frac{w^TS_Bw}{w^TS_Ww}.
\]

The optimal two-class projection direction is proportional to \(S_W^{-1}(m_1-m_2)\), when the inverse exists. **This is the projection derivation in the supplied LDA notes (pages 1–3).**

### 7.2 What `lda.py` actually implements

The class does **not** explicitly optimize the Fisher ratio or return a reduced-dimensional Fisher projection. It implements the related **Gaussian shared-covariance discriminant classifier**.

Class means and priors:

\[
\mu_c=\frac1{N_c}\sum_{i:y_i=c}x_i,\qquad P(c)=\frac{N_c}{N}.
\]

Pooled covariance estimate:

\[
\Sigma=\frac{\sum_c\sum_{i:y_i=c}(x_i-\mu_c)(x_i-\mu_c)^T}{N-C}.
\]

Regularization used by the code:

\[
\Sigma_{\mathrm{reg}}=(1-r)\Sigma+r\frac{\operatorname{tr}(\Sigma)}d I+10^{-5}I.
\]

Linear class score:

\[
\delta_c(x)=x^T\Sigma_{\mathrm{reg}}^{-1}\mu_c
-\frac12\mu_c^T\Sigma_{\mathrm{reg}}^{-1}\mu_c+\log P(c).
\]

Prediction: \(\widehat c=\arg\max_c\delta_c(x)\).

**Code:** `LDA(reg=0.05).fit(X_train,y_train)` computes class means, pooled covariance, regularized inverse action via `np.linalg.solve`, and offsets. `predict` compares scores. The distinction between Fisher projection and Gaussian LDA classification should be stated during viva.

## 8. K-Nearest Neighbours — `src/knn.py`

**Purpose:** Predict a label using nearby **labelled training observations**; no parametric distribution or linear boundary is assumed.

Squared Euclidean distance:

\[
d^2(x,x_i)=\sum_{j=1}^d(x_j-x_{ij})^2
=\|x\|^2+\|x_i\|^2-2x^Tx_i.
\]

Find the K closest training points, \(\mathcal N_K(x)\), and use majority voting:

\[
\widehat y=\arg\max_c\sum_{i\in\mathcal N_K(x)}\mathbf 1[y_i=c].
\]

**Code:** `KNN(k=5).fit(X_train,y_train)` stores training vectors and labels. `predict(X)` computes distances in chunks, uses `np.argpartition` to find the K closest indices, and votes. `chunk=128` limits temporary memory usage. Ties are resolved by the ordering of `np.unique` class labels (smallest class wins).

**Why compare with PCA:** Euclidean distances can become less informative when many dimensions are irrelevant; PCA may help or may discard class-separating information.

## 9. Support Vector Machine — `src/svm.py`

**Purpose:** Find a decision boundary with a large margin. The professor's SVM notes derive the hyperplane, geometric distance, canonical margin, primal/dual objectives, support vectors and kernels (pages 2–6).

### 9.1 Linear decision boundary and margin

\[
g(x)=w^Tx+b,\qquad \widehat y=\operatorname{sign}(g(x)).
\]

Distance from a point to the hyperplane:

\[
\frac{|w^Tx+b|}{\|w\|}.
\]

With labels \(y_i\in\{-1,+1\}\), the canonical hard-margin constraints are

\[
y_i(w^Tx_i+b)\ge 1.
\]

Hard-margin primal:

\[
\min_{w,b}\frac12\|w\|^2.
\]

The actual implementation uses **soft-margin** SVM with penalty C, which allows violations:

\[
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_i\xi_i,
\quad y_i(w^Tx_i+b)\ge1-\xi_i,\ \xi_i\ge0.
\]

### 9.2 Dual optimization and kernels

Dual objective:

\[
\max_{\alpha}\left[\sum_i\alpha_i-
\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jK(x_i,x_j)\right]
\]

subject to \(0\le\alpha_i\le C\) and \(\sum_i\alpha_iy_i=0\).

The implemented kernels are:

- **Linear:** \(K(x,z)=x^Tz\).
- **Gaussian/RBF:** \(K(x,z)=\exp[-\gamma\|x-z\|^2]\).

Decision function:

\[
f(x)=\sum_{i\in SV}\alpha_i y_iK(x_i,x)+b.
\]

**Code:** `BinarySVM` uses a simplified **Sequential Minimal Optimization (SMO)** procedure, updating two dual coefficients at a time under the constraints. It stores `support_`, `dual_`, and `b_`. `SVM` builds a **one-vs-rest** binary model per class and selects the class with the largest decision score.

**Important implementation limits:** The SVM wrapper deliberately subsamples training data when `len(X) > max_train` (the notebooks set 1500 for non-speech and 1000 for speech). The actual number is stored in `train_count_`. This means SVM may use **fewer training observations** than other classifiers; report that limitation. The simplified SMO optimizer also has iteration/pass limits and is not a full production SVM solver.

## 10. Speech feature representation — `src/speech.py`

Each TIDIGITS recording is a variable-length matrix \(X_i\in\mathbb R^{T_i\times39}\), with one acoustic feature vector per time frame.

For **LDA, KNN and SVM**, convert each recording into a fixed-length 78-dimensional vector:

\[
s_i=[\underbrace{\mu_1,\ldots,\mu_{39}}_{39\text{ frame means}},
\underbrace{\sigma_1,\ldots,\sigma_{39}}_{39\text{ frame standard deviations}}].
\]

For feature j:

\[
\mu_j=\frac1T\sum_{t=1}^T x_{tj},\qquad
\sigma_j=\sqrt{\frac1T\sum_{t=1}^T(x_{tj}-\mu_j)^2}.
\]

**Why:** Ordinary LDA/KNN/SVM require equal-length vectors; mean/std summarization makes recordings comparable. **Limitation:** temporal order is lost.

For **GMM**, keep the original sequence order. `FrameTransformer` supports applying a fitted scaler and optional PCA to every frame without changing sequence length; the notebook also performs these transforms directly.

## 11. Evaluation — `src/evaluation.py`

### Accuracy

\[
\mathrm{Accuracy}=\frac{\text{correct predictions}}{\text{total predictions}}.
\]

`accuracy(y_true, y_pred)` returns the fraction correctly classified.

### Confusion matrix

\[
M_{ij}=\#\{n:y_n=c_i,\ \widehat y_n=c_j\}.
\]

Rows are **true** classes; columns are **predicted** classes. Diagonal entries are correct predictions. `confusion(y_true, y_pred, labels)` computes this matrix directly with NumPy.

### Experimental protocol

1. Use the **same train/validation/test split** for original and PCA branches.
2. Fit scaling, PCA and classifiers using **training data only**.
3. Compare hyperparameters (K for PCA, GMM components, neighbours, SVM C/kernel) on **validation data**.
4. Report final held-out **test accuracy** and confusion matrices only after choosing settings.
5. Record the actual settings and any subsampling (especially SVM training subsample selection).

**Current notebook caveat:** The supplied notebooks print both validation and test results for fixed demonstration settings. For a strict unbiased final report, select settings from validation first, then run and report the held-out test evaluation once. The synthetic notebook's random split is not explicitly stratified; check class counts.

## 12. Which algorithm is used where?

| Method | Synthetic | MNIST | Image features | TIDIGITS |
|---|:---:|:---:|:---:|:---:|
| StandardScaler | ✓ | ✓ | ✓ | ✓ |
| PCA | ✓ | ✓ | ✓ | ✓ |
| GMM | ✓ | ✓ | ✓ | ✓ (frame-level) |
| Gaussian LDA classifier | ✓ | ✓ | ✓ | ✓ (78-D summary) |
| KNN | ✓ | ✓ | ✓ | ✓ (78-D summary) |
| SVM | ✓ | ✓ | ✓ | ✓ (78-D summary) |
| K-means | optional utility | optional utility | optional utility | optional utility |

**Notebook PCA choices:** Synthetic `K=8`; MNIST `K=40`; image features `K=12`; speech summaries `K=20`; speech frames `K=15`. These are the current code settings, **not values mandated by the assignment**.

## 13. Mathematical distinctions to remember for viva

| Comparison | Key distinction |
|---|---|
| K-means vs GMM | Hard nearest-centroid assignments vs soft probabilistic responsibilities; K-means clusters, GMM estimates a density. |
| GMM vs LDA | GMM models multiple Gaussian components per class; implemented LDA uses one Gaussian per class with a shared covariance. |
| Fisher LDA vs implemented LDA | Fisher LDA optimizes a projection scatter ratio; implemented LDA computes Gaussian discriminant scores. |
| KNN vs SVM | KNN stores training samples and votes locally; SVM learns a maximum-margin boundary represented by support vectors. |
| PCA vs LDA | PCA maximizes variance without labels; LDA uses class labels to discriminate. |
| Original vs PCA | Same classifier, but PCA changes the feature representation; accuracy may increase or decrease. |

## 14. Important limitations and honesty checklist

- GMM estimates **diagonal**, not full, covariance matrices; the notes give the general full-covariance update.
- `lda.py` is a **Gaussian LDA classifier**, not a separate implementation of Fisher projection.
- SVM uses simplified SMO, one-vs-rest multiclass scoring, and a **training subsample cap**.
- Speech methods use **different representations**, so their results cannot be attributed solely to algorithm choice.
- K-means is supplied as an independent utility; it is **not currently used to initialize GMM**.
- These are code formulas and intended procedures, **not claims that every experiment has been run or every result verified**.

## 15. Sources and mapping to notes

- **AI5707 Assignment 8:** Required classifiers, comparison of accuracy, and repeat with top-K PCA features.
- **`k_means_gmm(1).pdf`:** K-means hard assignments and centroid updates (pages 2–3); GMM mixture densities, responsibilities, E/M updates (pages 5–12).
- **`MLE_kmeans(1).pdf`:** MLE and multivariate Gaussian (pages 1–3); K-means distortion/centroids and mixtures (pages 10–12).
- **`LDA.pdf`:** Fisher's projected means, within/between scatter and Rayleigh quotient (pages 1–3). Gaussian shared-covariance discriminant implementation is related but **not the exact Fisher algorithm derived there**.
- **`SVM.pdf`:** Hyperplane, margin, primal/dual, support vectors and kernel trick (pages 2–6).
- **`overall_idea.pdf`:** Dataset-specific experiment design, speech representations, evaluation and leakage precautions.

**Note:** The uploaded handwritten notes do not supply the exact implementations of standardization, PCA, KNN, or the SMO optimization procedure. Those formulas and code descriptions are documented from the actual project modules and standard algorithm definitions, rather than attributed to the professor's handwritten notes.
