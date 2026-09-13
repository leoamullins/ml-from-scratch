# ml-from-scratch

Classic machine learning algorithms implemented from scratch in Python with NumPy and pandas, plus `cvxopt` to solve the SVM quadratic programs. They're packaged as an installable library, `mlscratch`, and each one has a Jupyter notebook demo. scikit-learn appears only in the demos (for example datasets) and the tests (as a reference to check results against); the library itself never imports it.

## Project structure

```text
ml-from-scratch/
├── mlscratch/                      # the library
│   ├── linear_regression.py        # LinearRegression: linear & polynomial regression
│   ├── logistic_regression.py      # LogisticRegression, MultinomialRegression: binary & multiclass
│   ├── svm.py                      # SVMHardMargin, SVMSoftMargin, SVM: primal & dual (kernel) SVMs
│   ├── tree.py                     # DecisionTreeID3: decision tree classifier (ID3)
│   ├── metrics.py                  # placeholder, not implemented yet
│   └── model_selection.py          # placeholder, not implemented yet
├── notebooks/                      # one demo notebook per algorithm
│   ├── linear_regression_demo.ipynb
│   ├── logistic_regression_demo.ipynb
│   ├── svm_demo.ipynb
│   └── tree_demo.ipynb
├── tests/
│   └── test_vs_sklearn.py          # checks each model against scikit-learn
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
- `notebooks`: `plot` plus `ipykernel` and scikit-learn, which the demo notebooks need.
- `test`: pytest and scikit-learn, for the tests.

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

## Validation against scikit-learn

[`tests/test_vs_sklearn.py`](tests/test_vs_sklearn.py) fits each model and its scikit-learn counterpart on the same data, prints how close they are, and checks that they agree.

```bash
pip install -e ".[test]"
pytest -s
```

The `-s` flag shows the numbers each test prints; without it pytest hides them. Current results:

| mlscratch | scikit-learn | Data | Result |
| --- | --- | --- | --- |
| `LinearRegression` | `LinearRegression()` | `make_regression`, 200 × 5 | coefficients match to 9.9e-14 |
| `LogisticRegression` | `LogisticRegression(C=np.inf)` (no penalty) | overlapping classes, 500 × 4 | coefficients match to 3.6e-9; 100% prediction agreement |
| `MultinomialRegression` | `LogisticRegression(C=1/(nλ))` | iris, 150 × 4 | coefficients match to 4.6e-7; 100% prediction agreement |
| `SVMHardMargin` | `SVC(kernel="linear", C=1e3)` | separable blobs, 100 × 2 | coefficients match to 2.9e-7 |
| `SVMSoftMargin` | `SVC(kernel="linear", C=1)` | overlapping classes, 200 × 4 | coefficients match to 1.7e-6; 100% prediction agreement |
| `SVM(kernel="linear")` | `SVC(kernel="linear", C=1)` | overlapping classes, 200 × 4 | 100% prediction agreement on 2,000 new points; 100% of support vectors shared; coefficients match to 1.7e-6 |
| `SVM(kernel="rbf")` | `SVC(kernel="rbf", C=1, gamma=1)` | `make_moons`, 200 × 2 | 100% prediction agreement on 2,000 new points; decision function matches to 3.3e-4; 98.4% of support vectors shared |
| `DecisionTreeID3` | `DecisionTreeClassifier(criterion="entropy", max_depth=6)` | 6 binary features, 10% label noise | 96.5% prediction agreement on 200 test rows; 100% outside tied leaves |

"Coefficients match to x" means the largest absolute difference between the weights. The tests also check that the intercepts match.

- **Logistic and multinomial regression** are run to convergence (`learning_rate=1.0`, `tol=0`). With the default `tol=1e-6` and iteration limits, gradient descent stops before it reaches the optimum.
- **Multinomial regression** minimises mean cross-entropy + λ/2 ‖W‖² on standardised features. scikit-learn minimises C × total cross-entropy + ½ ‖W‖², which is the same objective when C = 1/(nλ), so it's fitted on the same standardised features with that C.
- **The one RBF support vector that isn't shared** is an extra one in `SVM`. It lies on the edge of the margin, where its α should be exactly 0. cvxopt's solution leaves it at 2e-5, just above the 1e-5 cutoff, while scikit-learn's solver sets it to 0.
- **The tree** is compared on binary features, where ID3's one-branch-per-value split and scikit-learn's yes/no split are the same split. Every disagreement is in a leaf with equal class counts: scikit-learn predicts the smaller label there, and ID3 whichever label pandas lists first.

## Linear regression

`mlscratch/linear_regression.py` implements `LinearRegression`, fitted via the closed-form normal equation rather than gradient descent:

```text
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
- `DecisionTreeID3(features, target)` takes a dict mapping each feature to every value it can take, so the tree has a branch for every value even if a value never reaches a node during training; that branch predicts the majority class at that node.
- `fit(data)` takes a single table (anything `pd.DataFrame` accepts) holding the feature columns and the `target` column, rather than separate `X` and `y`. `predict(row)` classifies one row, given as a dict or a pandas row.
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
L = -\frac{1}{n} \sum_{i = 1}^{n} \sum_{k = 1}^{K} y_{ik} \log(p_{ik}) + \frac{\lambda}{2} \lVert W \rVert^2
$$

where λ is the `lam` hyperparameter, and minimises it with batch gradient descent. This outputs logits in `_scores` for each class and then the `_softmax` function is used in the core loop.

See [`notebooks/logistic_regression_demo.ipynb`](notebooks/logistic_regression_demo.ipynb) for a binary example on a synthetic 2D dataset and a multinomial example on the iris dataset.

## Support vector machines

`mlscratch/svm.py` implements three support vector machine classifiers, all solved as quadratic programs via `cvxopt`: `SVMHardMargin` and `SVMSoftMargin` solve the primal problem, and `SVM` solves the dual, which allows kernels.

A linear classifier predicts via `sign(w · x + b)`. The distance from a point to the separating hyperplane `w · x + b = 0` is `(w · x + b) / ‖w‖`, so scaling `w` and `b` up shrinks that distance without changing any prediction. SVMs pin this scale freedom down by requiring the closest points to sit at distance exactly `1/‖w‖`, i.e. `yᵢ(w · xᵢ + b) ≥ 1`. The margin — the gap between the two classes — is then `2/‖w‖`, so maximizing the margin is the same as minimizing `‖w‖`, which is what the two primal classifiers below solve for.

- `SVMHardMargin` solves the primal QP

  ```math
  \min \frac{1}{2} \| \textbf{w} \|_2 ^ 2 \hspace{10pt} \text{s.t.} \hspace{10pt} y_i(\textbf{w} x_i + b) \geq 1
  ```

  directly. This assumes the two classes are linearly separable — with no separating hyperplane, no `(w, b)` satisfies every constraint and the QP is infeasible.

- `SVMSoftMargin` handles non-separable data by adding a per-point slack $\xi_i$ that lets a point violate its margin, penalized by a cost `C`:

  ```math
  \min \frac{1}{2} \| \textbf{w} \|_2 ^ 2 + C \sum_{i=1}^N \xi_i
  ```

  `C` trades off margin width against how many points are allowed to be misclassified or fall inside the margin: large `C` penalizes slack heavily (behaving closer to hard-margin), small `C` tolerates more violations for a wider margin.

Both formulations are quadratic in `w` with linear inequality constraints, so `fit(X, y)` casts them into the standard QP form `min ½zᵀPz + qᵀz s.t. Gz ≤ h` — with `z = [w, b]` for the hard-margin case and `z = [w, b, ξ]` for the soft-margin case — and hands them to `cvxopt.solvers.qp`. `predict(X)` then returns `sign(w · X + b)`. Both classes expect labels of −1 and +1, so convert 0/1 labels first, e.g. with `np.where(y == 1, 1, -1)`.

### Dual problem & the kernel trick

`SVM` solves the same soft-margin problem as `SVMSoftMargin`, but in its **dual** form rather than the primal. Starting from the primal Lagrangian and eliminating `w`, `b`, and the slack variables via the KKT conditions leaves a QP purely in terms of the Lagrange multipliers `α`:

$$
\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac12 \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \hspace{10pt} \text{s.t.} \hspace{10pt} 0 \leq \alpha_i \leq C, \hspace{10pt} \sum_{i=1}^N \alpha_i y_i = 0
$$

The dual touches the data only through the dot products $x_i \cdot x_j$, never through an $x_i$ on its own. That's what enables the **kernel trick**: replace the dot product with any kernel function $K(x_i, x_j)$ that behaves like an inner product in some (possibly much higher-dimensional) feature space, and the same QP fits a nonlinear boundary without ever forming that feature space explicitly.

`SVM(kernel="linear", C=1.0, gamma=1.0)` follows the same `fit` / `predict` workflow as the rest of the library:

- `fit(X, y)` accepts any two class labels (0/1, say) and maps them to −1/+1 internally, since the dual objective and the constraint `Σ αᵢyᵢ = 0` both assume ±1. It raises a `ValueError` if `y` doesn't contain exactly two classes, then solves the dual with the chosen kernel (see below).
- `decision_function(X)` returns the raw score `f(x)`. Its sign is the predicted class; it is 0 on the decision boundary and ±1 on the edges of the margin. Use it when you need a continuous score, e.g. for ROC-AUC or for plotting the boundary.
- `predict(X)` maps the sign of `f(x)` back to the original labels: the larger of the two labels (in sorted order) where `f(x) ≥ 0`, the smaller one otherwise.

The `kernel` argument picks between:

- `"linear"` (solved by `fit_linear`): the plain dot product $K(x_i, x_j) = x_i \cdot x_j$. The dual is negated and cast into `cvxopt`'s `min ½zᵀPz + qᵀz s.t. Gz ≤ h, Az = b` form, with `z = α`, `P = diag(y) (XXᵀ) diag(y)`, the box constraint `0 ≤ α ≤ C` as `G, h`, and `Σ αᵢyᵢ = 0` as `A, b`. Once solved, `w = Σᵢ αᵢ yᵢ xᵢ`, and `b` is recovered by averaging `yᵢ − w · xᵢ` over the margin support vectors — points with `0 < αᵢ < C`, which by complementary slackness sit exactly on the margin. The score is then `f(x) = w · x + b`, the same as the primal solvers above.

- `"rbf"` (solved by `fit_RBF`): the **RBF (Gaussian) kernel**

  ```math
  K(x_i, x_j) = \exp\left(-\gamma \lVert x_i - x_j \rVert^2\right)
  ```

  computed by `_rbf` via the expansion $\lVert x_i - x_j \rVert^2 = \lVert x_i \rVert^2 + \lVert x_j \rVert^2 - 2\thinspace x_i \cdot x_j$, which avoids looping over pairs. This kernel implicitly maps each point into an infinite-dimensional feature space, so `w` can no longer be formed explicitly — everything has to stay expressed in terms of `α`, `y`, and the kernel. `b` is recovered as in the linear case, with the kernel sum in place of `w · xᵢ`, and `decision_function(X)` evaluates the score directly against the stored training points:

  ```math
  f(x) = \sum_{i=1}^N \alpha_i y_i K(x, x_i) + b
  ```

  Only the support vectors (`αᵢ > 0`) contribute to the sum. `γ` controls how tightly each support vector's influence is localized: small `γ` gives smooth, near-linear boundaries; large `γ` lets the boundary hug individual points, risking overfitting.

See [`notebooks/svm_demo.ipynb`](notebooks/svm_demo.ipynb) for the fitted decision boundary and margin (hard/soft-margin, linear-kernel cases), and a concentric-circles example where the RBF kernel carves out a closed nonlinear boundary that `kernel="linear"` can't fit.

## Status

Work in progress. More algorithms are planned, along with shared evaluation utilities in `mlscratch.metrics` and `mlscratch.model_selection`.
