# ml-from-scratch

Classic machine learning algorithms implemented from scratch in Python (NumPy/pandas only, no scikit-learn). They're packaged as an installable library, `mlscratch`, and each one has a Jupyter notebook demo.

## Project structure

```text
ml-from-scratch/
├── mlscratch/                      # the library
│   ├── linear_regression.py        # LinearRegression: linear & polynomial regression
│   ├── logistic_regression.py      # LogisticRegression: binary logistic regression
│   ├── svm.py                      # SVMHardMargin, SVMSoftMargin, SVM: primal & dual (kernel) SVMs
│   ├── tree.py                     # DecisionTreeID3: decision tree classifier (ID3)
│   ├── metrics.py                  # placeholder, not implemented yet
│   └── model_selection.py          # placeholder, not implemented yet
├── notebooks/                      # one demo notebook per algorithm
│   ├── linear_regression_demo.ipynb
│   ├── logistic_regression_demo.ipynb
│   ├── svm_demo.ipynb
│   └── tree_demo.ipynb
└── pyproject.toml                  # package metadata & dependencies
```

## Installation

Requires Python 3.10+. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks]"
```

The `-e` (editable) install makes `import mlscratch` load directly from the `mlscratch/` folder, so source edits are picked up after a kernel restart without reinstalling. The optional extras are:

- `plot`: matplotlib. The core library doesn't need it; only `DecisionTreeID3.plot()` uses it.
- `notebooks`: `plot` plus `ipykernel`, which the demo notebooks need.

Use `pip install -e .` for the core library alone (NumPy, pandas, cvxopt).

## Usage

Each algorithm lives in its own module:

```python
import numpy as np
from mlscratch.linear_regression import LinearRegression

X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.1, 4.9, 7.2, 8.8])

model = LinearRegression()
model.fit(X, y)
model.predict(np.array([5.0]))  # array([10.85])
```

To run the demos, open any notebook in [`notebooks/`](notebooks/) in VS Code or another Jupyter frontend, and select the `.venv` kernel. To use Jupyter in the browser, run `pip install jupyterlab` and then `jupyter lab notebooks/`.

## Linear regression

`mlscratch/linear_regression.py` implements `LinearRegression`, fitted via the closed-form normal equation rather than gradient descent:

```
θ = (XᵀX)⁻¹Xᵀy
```

- `fit(X, y)` — solves for weights and bias directly.
- `predict(X)` — applies the learned weights.
- `fit_poly(X, y, degree)` / `predict_poly(X)` — expands `X` into polynomial features (`x, x², …, x^degree`) via `poly_features`, then reuses the same linear solver. This is how polynomial regression is achieved without a separate algorithm: it's still linear regression, just on transformed features.

See [`notebooks/linear_regression_demo.ipynb`](notebooks/linear_regression_demo.ipynb) for a straight-line fit and a cubic fit.

## Decision tree (ID3)

`mlscratch/tree.py` implements `DecisionTreeID3`, a decision tree classifier for categorical features built using Quinlan's ID3 algorithm.

- **Entropy** (`_entropy`) measures the impurity of a set of labels.
- **Information gain** (`gain`) measures how much splitting on a feature reduces entropy.
- **`id3`** recursively builds the tree: at each step it picks the feature with the highest information gain (`best_gain`), splits the data by that feature's values, and recurses. Recursion stops when a subset is pure (a `Leaf`) or there are no features left to split on (falls back to the majority class).
- The tree is made of two node types: `Node` (an internal split on a feature) and `Leaf` (a class label).
- `plot()` gives a simple matplotlib visualisation of the fitted tree (requires the `plot` extra).

See [`notebooks/tree_demo.ipynb`](notebooks/tree_demo.ipynb) for the canonical "play tennis" example.

## Logistic regression

`mlscratch/logistic_regression.py` implements `LogisticRegression`, a binary classifier fitted by maximizing log-likelihood via batch gradient descent and `MultinomialRegression`, a classifier fitted with batch gradient descent.

- `sigmoid(z)` — the logistic function, squashes `z` into `(0, 1)`.
- `gradient_descent_log_likelihood` — runs gradient descent for `num_steps`, updating `w` and `b` at each step (`update_w_and_b_log_likelihood`) and logging binary cross-entropy loss.
- `fit(X, y)` — learns `w` and `b` from training data, storing the loss history.
- `predict(X)` — applies the sigmoid to `w · X + b` and thresholds at `positive_threshold`.

### Multinomial Regression

This method computes the loss as regularised multinomial cross entropy
$$
L = -\frac{1}{n} \sum_{i = 1}^{n} \sum_{k = 1}^{K} y_{ik} \log(p_{ik})
$$

with batch gradient descent. This outputs logits in `_scores` for each class and then the `_softmax` function is used in the core loop.

See [`notebooks/logistic_regression_demo.ipynb`](notebooks/logistic_regression_demo.ipynb) for the training curve and decision boundary.

## Support vector machines

`mlscratch/svm.py` implements two support vector machine classifiers, both solved as quadratic programs via `cvxopt`.

A linear classifier predicts via `sign(w · x + b)`. The distance from a point to the separating hyperplane `w · x + b = 0` is `(w · x + b) / ‖w‖`, so scaling `w` and `b` up shrinks that distance without changing any prediction. SVMs pin this scale freedom down by requiring the closest points to sit at distance exactly `1/‖w‖`, i.e. `yᵢ(w · xᵢ + b) ≥ 1`. The margin — the gap between the two classes — is then `2/‖w‖`, so maximizing the margin is the same as minimizing `‖w‖`, which is what both classifiers below solve for.

- `SVMHardMargin` solves the primal QP

  $$
  \min \frac{1}{2} \| \textbf{w} \|_2 ^ 2 \hspace{10pt} \text{s.t.} \hspace{10pt} y_i(\textbf{w} x_i + b) \geq 1
  $$

  directly. This assumes the two classes are linearly separable — with no separating hyperplane, no `(w, b)` satisfies every constraint and the QP is infeasible.

- `SVMSoftMargin` handles non-separable data by adding a per-point slack $\xi_i$ that lets a point violate its margin, penalized by a cost `C`:

$$
  \min \frac{1}{2} \| \textbf{w} \|_2 ^ 2 + C \sum_{i=1}^N \xi_i
$$

  `C` trades off margin width against how many points are allowed to be misclassified or fall inside the margin: large `C` penalizes slack heavily (behaving closer to hard-margin), small `C` tolerates more violations for a wider margin.

Both formulations are quadratic in `w` with linear inequality constraints, so `fit(X, y)` casts them into the standard QP form `min ½zᵀPz + qᵀz s.t. Gz ≤ h` — with `z = [w, b]` for the hard-margin case and `z = [w, b, ξ]` for the soft-margin case — and hands them to `cvxopt.solvers.qp`. `predict(X)` then returns `sign(w · X + b)`.

### Dual problem & the kernel trick

`SVM` solves the same soft-margin problem as `SVMSoftMargin`, but in its **dual** form rather than the primal. Starting from the primal Lagrangian and eliminating `w`, `b`, and the slack variables via the KKT conditions leaves a QP purely in terms of the Lagrange multipliers `α`:

$$
\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac12 \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \hspace{10pt} \text{s.t.} \hspace{10pt} 0 \leq \alpha_i \leq C, \hspace{10pt} \sum_{i=1}^N \alpha_i y_i = 0
$$

The dual touches the data only through the dot products $x_i \cdot x_j$, never through an $x_i$ on its own. That's what enables the **kernel trick**: replace the dot product with any kernel function $K(x_i, x_j)$ that behaves like an inner product in some (possibly much higher-dimensional) feature space, and the same QP fits a nonlinear boundary without ever forming that feature space explicitly.

- `fit_linear(X, y)` solves the dual with the plain linear kernel $K(x_i, x_j) = x_i \cdot x_j$. It's negated and cast into `cvxopt`'s `min ½zᵀPz + qᵀz s.t. Gz ≤ h, Az = b` form, with `z = α`, `P = diag(y) (XXᵀ) diag(y)`, the box constraint `0 ≤ α ≤ C` as `G, h`, and `Σ αᵢyᵢ = 0` as `A, b`. Once solved, `w = Σᵢ αᵢ yᵢ xᵢ`, and `b` is recovered by averaging `yᵢ − w · xᵢ` over the margin support vectors — points with `0 < αᵢ < C`, which by complementary slackness sit exactly on the margin. `predict_linear(X)` then returns `sign(w · X + b)`, same as the primal solvers above.

- `fit_RBF(X, y)` swaps in the **RBF (Gaussian) kernel**

  $$
  K(x_i, x_j) = \exp\left(-\gamma \lVert x_i - x_j \rVert^2\right)
  $$

  computed by `_rbf` via the expansion $\lVert x_i - x_j \rVert^2 = \lVert x_i \rVert^2 + \lVert x_j \rVert^2 - 2\, x_i \cdot x_j$, which avoids looping over pairs. This kernel implicitly maps each point into an infinite-dimensional feature space, so `w` can no longer be formed explicitly — everything has to stay expressed in terms of `α`, `y`, and the kernel. `predict_RBF(X)` therefore evaluates the decision function directly against the stored training points (the support vectors):

  $$
  f(x) = \text{sign}\left(\sum_{i=1}^N \alpha_i y_i K(x, x_i) + b\right)
  $$

  `γ` controls how tightly each support vector's influence is localized: small `γ` gives smooth, near-linear boundaries; large `γ` lets the boundary hug individual points, risking overfitting.

See [`notebooks/svm_demo.ipynb`](notebooks/svm_demo.ipynb) for the fitted decision boundary and margin (hard/soft-margin, linear-kernel cases), and a concentric-circles example where the RBF kernel carves out a closed nonlinear boundary that the linear dual solver can't fit.

## Status

Work in progress. More algorithms are planned, along with shared evaluation utilities in `mlscratch.metrics` and `mlscratch.model_selection`.
