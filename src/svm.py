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

        # random  value
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(d + 1)          # son element = bias

        # 4) Pegasos dövrü
        for t in range(1, self.n_iters + 1):
            i = rng.integers(n)                      # təsadüfi bir nümunə
            eta = 1.0 / (self.lambda_ * t)           # öyrənmə sürəti
            margin = y_pm[i] * (Xb[i] @ w)           # köhnə w ilə marja
            w *= (1 - eta * self.lambda_)            # həmişə: w-ni kiçilt
            if margin < 1:                           # səhvdir və ya sərhədə yaxındır
                w += eta * y_pm[i] * Xb[i]           # w-ni bu nümunəyə doğru it

        
        self.w = w[:-1]
        self.b = w[-1]
        return self

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        raise NotImplementedError("TODO: decision_function")

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        raise NotImplementedError("TODO: predict")
