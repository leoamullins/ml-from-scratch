# ml-from-scratch

Classic machine learning algorithms implemented from scratch in Python (NumPy/pandas only, no scikit-learn), with a Jupyter notebook demoing each one.

## Contents

- [`linear-regression/`](linear-regression/) — linear & polynomial regression
- [`decision-tree-ID3/`](decision-tree-ID3/) — decision tree classifier (ID3)
- [`logistic-regression/`](logistic-regression/) — binary logistic regression
- [`svm/`](svm/) — hard-margin & soft-margin support vector machines

## linear-regression

`model.py` implements `LinearRegression`, fitted via the closed-form normal equation rather than gradient descent:

```
θ = (XᵀX)⁻¹Xᵀy
```

- `fit(X, y)` — solves for weights and bias directly.
- `predict(X)` — applies the learned weights.
- `fit_poly(X, y, degree)` / `predict_poly(X)` — expands `X` into polynomial features (`x, x², …, x^degree`) via `poly_features`, then reuses the same linear solver. This is how polynomial regression is achieved without a separate algorithm: it's still linear regression, just on transformed features.

See `demo.ipynb` for a straight-line fit and a cubic fit.

## decision-tree-ID3

`model.py` implements `DecisionTreeID3`, a decision tree classifier for categorical features built using Quinlan's ID3 algorithm.

- **Entropy** (`_entropy`) measures the impurity of a set of labels.
- **Information gain** (`gain`) measures how much splitting on a feature reduces entropy.
- **`id3`** recursively builds the tree: at each step it picks the feature with the highest information gain (`best_gain`), splits the data by that feature's values, and recurses. Recursion stops when a subset is pure (a `Leaf`) or there are no features left to split on (falls back to the majority class).
- The tree is made of two node types: `Node` (an internal split on a feature) and `Leaf` (a class label).
- `plot()` gives a simple matplotlib visualisation of the fitted tree.

See `demo.ipynb` for the canonical "play tennis" example.

## logistic-regression

`model.py` implements `LogisticRegression`, a binary classifier fitted by maximizing log-likelihood via batch gradient descent.

- `sigmoid(z)` — the logistic function, squashes `z` into `(0, 1)`.
- `gradient_descent_log_likelihood` — runs gradient descent for `num_steps`, updating `w` and `b` at each step (`update_w_and_b_log_likelihood`) and logging binary cross-entropy loss.
- `fit(X, y)` — learns `w` and `b` from training data, storing the loss history.
- `predict(X)` — applies the sigmoid to `w · X + b` and thresholds at `positive_threshold`.

See `demo.ipynb` for the training curve and decision boundary.

## svm

`model.py` implements two support vector machine classifiers, both solved as quadratic programs via `cvxopt`.

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

See `demo.ipynb` for the fitted decision boundary and margin.

## Usage

Each folder is self-contained — `cd` into it and open the notebook:

```
cd linear-regression && jupyter notebook demo.ipynb
```

## Status

Work in progress. More algorithms to follow.
