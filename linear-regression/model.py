import numpy as np


class LinearRegression:
    def __init__(self) -> None:
        "initialise parameters with None"
        self.weights: np.ndarray = None
        self.bias: float = None
        self.degree: int = None

    def _init_params(self, n_features: int) -> None:
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        "fit the model, closed form"
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        params = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y
        self.bias = params[0]
        self.weights = params[1:]

    def fit_poly(self, X: np.ndarray, y: np.ndarray, degree: int = 2) -> None:
        X_poly = self.poly_features(X, degree)
        self.degree = degree

        self.fit(X_poly, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        "predict from a list of inputs"
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.weights + self.bias

    def predict_poly(self, X: np.ndarray) -> np.ndarray:
        X_poly = self.poly_features(X, self.degree)

        return self.predict(X_poly)

    @staticmethod
    def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
        "create poly features"
        return np.column_stack([x**d for d in range(1, degree + 1)])
