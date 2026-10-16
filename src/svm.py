"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:

    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0               # bias

    @staticmethod
    
    def _to_pm1(y, classes):
        y = np.asarray(y)
        return np.where(y == classes[1], 1.0, -1.0)

    def _objective(self, Xb, y_pm, w):
        margins = y_pm * (Xb @ w)
        return 0.5 * self.lambda_ * (w @ w) + np.mean(np.maximum(0.0, 1.0 - margins))

    def fit(self, X, y):
        # {-1,1}
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Need exactly 2 classes")
        self.classes_ = classes
        y_pm = self._to_pm1(y, self.classes_)

        # added 1 to x for bias
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])

        # random w value
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(d + 1)   
        self.history_ = []
        log_every = max(1, self.n_iters // 100)

        # w
        for t in range(1, self.n_iters + 1):
            i = rng.integers(n)                      
            eta = 1.0 / (self.lambda_ * t)           
            margin = y_pm[i] * (Xb[i] @ w)           
            w *= (1 - eta * self.lambda_)           
            if margin < 1:                           
                w += eta * y_pm[i] * Xb[i]   
            if t % log_every == 0:
                self.history_.append((t, self._objective(Xb, y_pm, w)))        

        
        self.w = w[:-1]
        self.b = w[-1]
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]
    def margin_width(self):
        return 2.0 / np.linalg.norm(self.w)

    def support_vector_mask(self, X, y):
        y_pm = self._to_pm1(y, self.classes_)
        margins = y_pm * self.decision_function(X)
        return margins <= 1


class RandomFourierFeatures:
    def __init__(self, n_components=200, gamma=0.1, random_state=42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.b = rng.uniform(0.0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W + self.b)