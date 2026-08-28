# ml-from-scratch

Classic machine learning algorithms implemented from scratch in Python (NumPy/pandas only, no scikit-learn), with a Jupyter notebook demoing each one.

## Contents

- [`linear-regression/`](linear-regression/) — linear & polynomial regression
- [`decision-tree-ID3/`](decision-tree-ID3/) — decision tree classifier (ID3)

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

## Usage

Each folder is self-contained — `cd` into it and open the notebook:

```
cd linear-regression && jupyter notebook demo.ipynb
```

## Status

Work in progress. More algorithms to follow.
