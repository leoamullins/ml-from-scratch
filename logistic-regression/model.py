import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def update_w_and_b_log_likelihood(X, y, w, b, alpha):
    n = X.shape[0]
    z = np.dot(X, w) + b
    p = sigmoid(z)
    dJ_dw = (X.T @ (p - y)) / n
    dJ_db = (p - y).sum() / n

    w -= alpha * dJ_dw
    b -= alpha * dJ_db
    return w, b


def gradient_descent_log_likelihood(X, y, alpha, num_steps):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    history = []

    for i in range(num_steps):
        w, b = update_w_and_b_log_likelihood(X, y, w, b, alpha)

        p = sigmoid(X @ w + b)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        history.append(loss)

        if i % 100 == 0:
            print(f"Epoch = {i}, Loss = {loss}")

    return w, b, history


class LogisticRegression:
    def __init__(self, alpha, positive_threshold, num_steps):
        self.loss = 0.0
        self.alpha = alpha
        self.num_steps = num_steps
        self.threshold = positive_threshold

    def fit(self, X, y):
        self.X = X
        self.y = y

        w, b, history = gradient_descent_log_likelihood(
            X, y, self.alpha, self.num_steps
        )
        self.w, self.b = w, b
        self.loss_history = history
        self.loss = history[-1]

    def predict(self, X):
        z = self.w @ X + self.b
        p = sigmoid(z)

        return 1 if p >= self.threshold else 0
