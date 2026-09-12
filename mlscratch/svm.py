import numpy as np
import cvxopt as cvx


class SVMHardMargin:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        P = np.zeros((n_features + 1, n_features + 1))
        P[0:-1, 0:-1] = np.eye(n_features)

        q = np.zeros(n_features + 1)
        X_tilde = np.hstack([X, np.ones((n_samples, 1))])
        G = -y[:, None] * X_tilde

        h = -np.ones(n_samples)

        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)

        cvx.solvers.options["show_progress"] = False
        sol = cvx.solvers.qp(P, q, G, h)

        if sol["status"] != "optimal":
            raise ValueError(f"QP did not converge: status = {sol['status']}")

        z = np.array(sol["x"]).flatten()
        self.w = z[:n_features]
        self.b = z[n_features]

    def predict(self, X):
        return np.sign(X @ self.w + self.b)


class SVMSoftMargin:
    def __init__(self, C=1.0):
        self.w = None
        self.b = None
        self.c = C

    def fit(self, X, y):
        n, d = X.shape
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        P = np.zeros((d + n + 1, d + n + 1))
        P[0:d, 0:d] = np.eye(
            d
        )  # quadratic term is only the w, so zeros everywhere except the w block

        q = np.zeros(d + 1 + n)
        q[d + 1 :] = self.c  # linear slack term

        X_tilde = np.hstack([X, np.ones((n, 1))])
        G = np.zeros((2 * n, d + 1 + n))
        G[0:n, 0 : d + 1] = -y[:, None] * X_tilde
        G[:n, d + 1 :] = -np.eye(n)
        G[n:, d + 1 :] = -np.eye(n)

        h = np.concatenate([-np.ones(n), np.zeros(n)])

        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)

        cvx.solvers.options["show_progress"] = False
        sol = cvx.solvers.qp(P, q, G, h)

        if sol["status"] != "optimal":
            raise ValueError(f"QP did not converge: status = {sol['status']}")

        z = np.array(sol["x"]).flatten()
        self.w = z[:d]
        self.b = z[d]

    def predict(self, X):
        return np.sign(X @ self.w + self.b)


class SVM:
    def __init__(self, kernel="linear", C=1.0, gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM needs exaclty 2 classes.")
        y_pm = np.where(y == self.classes_[1], 1.0, -1.0)
        if self.kernel == "linear":
            self.fit_linear(X, y_pm)
        elif self.kernel == "rbf":
            self.fit_RBF(X, y_pm)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def decision_function(self, X):
        if self.kernel == "linear":
            return X @ self.w + self.b
        return self._rbf(X, self.X_train) @ (self.alpha * self.y_train) + self.b

    def fit_linear(self, X, y):
        n, d = X.shape  # d is n_features and n is n_samples, X is (n, d) matrix
        X = X.astype(np.float64)
        y = y.astype(np.float64)  # cvx compatible

        P = np.diag(y) @ X @ X.T @ np.diag(y)

        q = -np.ones(n)

        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.concatenate([np.zeros(n), self.C * np.ones(n)])

        A = y.reshape(1, -1)
        b = 0.0

        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(A)
        b = cvx.matrix(b)

        cvx.solvers.options["show_progress"] = False
        sol = cvx.solvers.qp(P, q, G, h, A, b)

        if sol["status"] != "optimal":
            raise ValueError(f"QP did not converge: status = {sol['status']}")

        self.alpha = np.array(sol["x"]).flatten()

        margin_sv = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)

        self.w = ((self.alpha * y)[:, None] * X).sum(axis=0)
        self.b = np.mean(y[margin_sv] - X[margin_sv] @ self.w)

    def fit_RBF(self, X, y):
        n, d = X.shape  # d is n_features and n is n_samples, X is (n, d) matrix
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        self.X_train = X
        self.y_train = y

        K = self._rbf(X, X)
        P = np.diag(y) @ K @ np.diag(y)

        q = -np.ones(n)

        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.concatenate([np.zeros(n), self.C * np.ones(n)])

        A = y.reshape(1, -1)
        b = 0.0

        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(A)
        b = cvx.matrix(b)

        cvx.solvers.options["show_progress"] = False
        sol = cvx.solvers.qp(P, q, G, h, A, b)

        if sol["status"] != "optimal":
            raise ValueError(f"QP did not converge: status = {sol['status']}")

        self.alpha = np.array(sol["x"]).flatten()
        margin_sv = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)

        decision = K @ (self.alpha * y)
        self.b = np.mean(y[margin_sv] - decision[margin_sv])

    def predict(self, X):
        return np.where(
            self.decision_function(X) >= 0, self.classes_[1], self.classes_[0]
        )

    def _rbf(self, A, B):
        sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma * sq)
