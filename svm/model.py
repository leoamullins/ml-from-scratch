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
