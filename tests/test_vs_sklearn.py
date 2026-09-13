import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model
from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mlscratch.linear_regression import LinearRegression
from mlscratch.logistic_regression import LogisticRegression, MultinomialRegression
from mlscratch.svm import SVM, SVMHardMargin, SVMSoftMargin
from mlscratch.tree import DecisionTreeID3

# Each test fits one of my models and the matching scikit-learn model on the same
# data, prints how close they are, and checks that they agree.
# Run `pytest -s` to see the printed numbers.


def agreement(a, b):
    """Fraction of predictions that are the same."""
    return np.mean(np.asarray(a) == np.asarray(b))


def overlapping_classes(n_samples):
    """two non linearly separable classes"""
    return make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        class_sep=0.8,
        flip_y=0.05,
        random_state=0,
    )


def new_points(X, n=2000):
    """Random points spread over the same range as X."""
    rng = np.random.default_rng(1)
    return rng.uniform(X.min(axis=0), X.max(axis=0), size=(n, X.shape[1]))


def shared_support_vectors(mine, ref):
    """Fraction of support vectors that both SVMs picked."""
    mine_sv = np.flatnonzero(mine.alpha > 1e-5)  # the same cutoff SVM.fit uses
    shared = np.intersect1d(mine_sv, ref.support_)
    either = np.union1d(mine_sv, ref.support_)
    return len(shared) / len(either)


def test_linear_regression():
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=0)
    mine = LinearRegression()
    mine.fit(X, y)

    ref = linear_model.LinearRegression().fit(X, y)

    # Both solve for the weights directly, so they should match almost exactly.
    coef_diff = np.abs(mine.weights - ref.coef_).max()
    print(f"LinearRegression max coef diff: {coef_diff:.1e}")
    assert mine.weights == pytest.approx(ref.coef_, abs=1e-8)
    assert mine.bias == pytest.approx(ref.intercept_, abs=1e-8)


def test_logistic_regression():
    # testing overlapping classes
    X, y = overlapping_classes(500)

    # tol=0 turns off early stopping, so gradient descent runs all 5000 epochs.
    mine = LogisticRegression(learning_rate=1.0, num_epochs=5000, tol=0).fit(X, y)
    # C=np.inf means no penalty, the same loss as mine. max_iter is only a limit:
    # sklearn stops by itself after ~11 iterations.
    ref = linear_model.LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(X, y)

    # my regression model learns w and b on standardised features so I converted them back here
    coef = mine.w / mine.std_
    intercept = mine.b - np.sum(mine.w * mine.mean_ / mine.std_)

    coef_diff = np.abs(coef - ref.coef_[0]).max()
    agree = agreement(mine.predict(X), ref.predict(X))
    print(f"LogisticRegression max coef diff: {coef_diff:.1e}, agreement: {agree:.1%}")
    assert coef == pytest.approx(ref.coef_[0], abs=1e-6)
    assert intercept == pytest.approx(ref.intercept_[0], abs=1e-6)
    assert agree >= 0.99


def test_multinomial_regression():
    X, y = load_iris(return_X_y=True)
    lam = 0.01

    # tol=0 turns off early stopping, so gradient descent runs all 2000 iterations.
    mine = MultinomialRegression(
        learning_rate=1.0, max_iterations=2000, lam=lam, tol=0
    ).fit(X, y)
    # My model standardises the features and adds a lam/2 * ||W||^2 penalty.
    # sklearn adds the same penalty when C = 1 / (n * lam), so fit it with that C
    # on the same standardised features.
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    ref = linear_model.LogisticRegression(
        C=1 / (len(X) * lam), tol=1e-12, max_iter=10_000
    ).fit(X_std, y)

    # sklearn stores the weights as (classes, features), mine as (features, classes).
    coef_diff = np.abs(mine.W - ref.coef_.T).max()
    agree = agreement(mine.predict(X), ref.predict(X_std))
    print(
        f"MultinomialRegression max coef diff: {coef_diff:.1e}, agreement: {agree:.1%}"
    )
    assert mine.W == pytest.approx(ref.coef_.T, abs=1e-5)
    assert mine.b == pytest.approx(ref.intercept_, abs=1e-5)
    assert agree >= 0.99


def test_svm_hard_margin():
    # Two blobs far apart, so a straight line can separate them.
    X, y = make_blobs(
        n_samples=100, centers=[(-2, -2), (2, 2)], cluster_std=1.0, random_state=0
    )
    y = np.where(y == 1, 1, -1)  # SVMHardMargin needs labels -1 and +1

    mine = SVMHardMargin()
    mine.fit(X, y)
    # sklearn has no hard-margin SVM. A large C makes crossing the margin so costly
    # that it gives the hard-margin answer.
    # tol=1e-6 (default 1e-3) makes sklearn's answer more exact, here and below.
    ref = SVC(kernel="linear", C=1e3, tol=1e-6).fit(X, y)

    coef_diff = np.abs(mine.w - ref.coef_[0]).max()
    print(f"SVMHardMargin max coef diff: {coef_diff:.1e}")
    assert mine.w == pytest.approx(ref.coef_[0], abs=1e-4)
    assert mine.b == pytest.approx(ref.intercept_[0], abs=1e-4)


def test_svm_soft_margin():
    X, y = overlapping_classes(200)
    y = np.where(y == 1, 1, -1)  # SVMSoftMargin needs labels -1 and +1

    mine = SVMSoftMargin(C=1.0)
    mine.fit(X, y)
    ref = SVC(kernel="linear", C=1.0, tol=1e-6).fit(X, y)

    coef_diff = np.abs(mine.w - ref.coef_[0]).max()
    agree = agreement(mine.predict(X), ref.predict(X))
    print(f"SVMSoftMargin max coef diff: {coef_diff:.1e}, agreement: {agree:.1%}")
    assert mine.w == pytest.approx(ref.coef_[0], abs=1e-4)
    assert mine.b == pytest.approx(ref.intercept_[0], abs=1e-4)
    assert agree >= 0.99


def test_svm_linear():
    X, y = overlapping_classes(200)

    mine = SVM(kernel="linear", C=1.0)
    mine.fit(X, y)
    ref = SVC(kernel="linear", C=1.0, tol=1e-6).fit(X, y)

    # Check predictions on new points too, not just the training data.
    coef_diff = np.abs(mine.w - ref.coef_[0]).max()
    X_new = new_points(X)
    agree = agreement(mine.predict(X_new), ref.predict(X_new))
    shared = shared_support_vectors(mine, ref)
    print(f"SVM linear max coef diff: {coef_diff:.1e}, agreement: {agree:.1%}")
    print(f"SVM linear support vectors shared: {shared:.1%}")
    assert mine.w == pytest.approx(ref.coef_[0], abs=1e-4)
    assert mine.b == pytest.approx(ref.intercept_[0], abs=1e-4)
    assert agree >= 0.99
    assert shared >= 0.95


def test_svm_rbf():
    X, y = make_moons(n_samples=200, noise=0.25, random_state=0)

    mine = SVM(kernel="rbf", C=1.0, gamma=1.0)
    mine.fit(X, y)
    ref = SVC(kernel="rbf", C=1.0, gamma=1.0, tol=1e-6).fit(X, y)

    # An RBF model has no w to compare, so compare predictions on new points and
    # the raw scores from decision_function instead.
    X_new = new_points(X)
    agree = agreement(mine.predict(X_new), ref.predict(X_new))
    score_diff = np.abs(mine.decision_function(X) - ref.decision_function(X)).max()
    shared = shared_support_vectors(mine, ref)
    print(f"SVM rbf max score diff: {score_diff:.1e}, agreement: {agree:.1%}")
    print(f"SVM rbf support vectors shared: {shared:.1%}")
    assert agree >= 0.99
    assert score_diff < 1e-3
    assert shared >= 0.95


def test_decision_tree():
    # Random 0/1 features. With only two values per feature, ID3's split and
    # sklearn's split are the same, so the two trees should make the same choices.
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(700, 6))
    y = ((X[:, 0] == 1) & (X[:, 1] == 1)) | (X[:, 2] == 1)
    y = np.where(rng.random(700) < 0.1, ~y, y).astype(int)  # flip 10% of labels
    X_train, y_train, X_test = X[:500], y[:500], X[500:]

    # DecisionTreeID3 takes one table with named columns, target column included.
    columns = [f"x{i}" for i in range(6)]
    mine = DecisionTreeID3({c: [0, 1] for c in columns}, target="label")
    mine.fit(pd.DataFrame(X_train, columns=columns).assign(label=y_train))
    # ID3 uses each feature at most once on a path, so its depth is at most 6.
    ref = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=0)
    ref.fit(X_train, y_train)

    # predict takes one row at a time, so apply it to each row.
    test_table = pd.DataFrame(X_test, columns=columns)
    mine_pred = test_table.apply(mine.predict, axis=1).to_numpy()
    ref_pred = ref.predict(X_test)

    # Some leaves hold equal numbers of 0s and 1s, and the two libraries break those
    # ties differently. Everywhere else the predictions should match exactly.
    leaf_values = ref.tree_.value[ref.apply(X_test), 0]
    tied = leaf_values[:, 0] == leaf_values[:, 1]

    agree = agreement(mine_pred, ref_pred)
    agree_untied = agreement(mine_pred[~tied], ref_pred[~tied])
    print(f"DecisionTreeID3 agreement: {agree:.1%}, outside ties: {agree_untied:.1%}")
    assert agree >= 0.95
    assert agree_untied == 1.0
