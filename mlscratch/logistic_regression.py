import numpy as np
import pandas as pd


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def update_w_and_b_log_likelihood(X, y, w, b, alpha):
    n = X.shape[0]
    z = np.dot(X, w) + b
    p = sigmoid(z)
    dJ_dw = (X.T @ (p - y)) / n
    dJ_db = (p - y).sum() / n

    w = w - alpha * dJ_dw
    b -= alpha * dJ_db
    return w, b


def gradient_descent_log_likelihood(X, y, alpha, num_steps, tol, verbose=False):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    history = []
    prev_loss = None

    for i in range(num_steps):
        p = sigmoid(X @ w + b)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        history.append(loss)

        w, b = update_w_and_b_log_likelihood(X, y, w, b, alpha)

        if i % 100 == 0 and verbose:
            print(f"Epoch = {i}, Loss = {loss}")

        if prev_loss is not None and abs(prev_loss - loss) < tol * (1 + abs(loss)):
            break
        prev_loss = loss

    return w, b, history


class LogisticRegression:
    def __init__(
        self, learning_rate=0.1, positive_threshold=0.5, num_epochs=1000, tol=1e-6
    ):
        self.alpha = learning_rate
        self.num_steps = num_epochs
        self.threshold = positive_threshold
        self.tol = tol

    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)

        Xs = (X - self.mean_) / self.std_
        w, b, history = gradient_descent_log_likelihood(
            Xs, y, self.alpha, self.num_steps, self.tol
        )
        self.w, self.b = w, b
        self.loss_history = history
        self.loss = history[-1]
        return self

    def predict(self, X):
        Xs = (X - self.mean_) / self.std_

        z = Xs @ self.w + self.b
        p = sigmoid(z)

        return (p >= self.threshold).astype(int)

    def predict_proba(self, X):
        Xs = (X - self.mean_) / self.std_

        z = Xs @ self.w + self.b
        p1 = sigmoid(z)
        p0 = 1 - p1

        return np.column_stack([p0, p1])


class MultinomialRegression:
    def __init__(self, learning_rate=0.1, max_iterations=1000, lam=0.01, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.lam = lam
        self.tol = tol

        # fitted parameters
        self.W = None
        self.b = None
        self.classes_ = None
        self.mean_ = None
        self.std_ = None
        self.loss_history_ = None

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)  # avoid overflows
        E = np.exp(Z)

        return E / E.sum(axis=1, keepdims=True)

    def _scores(self, X_std):
        return X_std @ self.W + self.b

    def _loss(self, P, Y):
        data = -np.mean(np.sum(Y * np.log(P + 1e-15), axis=1))
        reg = 0.5 * self.lam * np.sum(self.W**2)
        return data + reg

    def fit(self, X, y):
        n, d = X.shape

        self.loss_history_ = []

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        Y = np.eye(K)[y_idx]

        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)  # to avoid division by 0
        Xs = (X - self.mean_) / self.std_

        self.W = np.zeros((d, K))
        self.b = np.zeros(K)

        prev_loss = None
        self.n_iter_ = 0

        for _ in range(self.max_iterations):
            Z = self._scores(Xs)
            P = self._softmax(Z)
            loss = self._loss(P, Y)
            self.loss_history_.append(loss)

            R = P - Y
            dW = Xs.T @ R / n + self.lam * self.W
            db = R.mean(axis=0)

            self.n_iter_ += 1
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            if prev_loss is not None and abs(prev_loss - loss) < self.tol * (
                1 + abs(loss)
            ):
                break
            prev_loss = loss
        return self

    def predict_proba(self, X):
        Xs = (X - self.mean_) / self.std_
        Z = self._scores(Xs)
        softmax = self._softmax(Z=Z)
        return softmax

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]
