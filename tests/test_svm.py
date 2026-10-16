import numpy as np
from src.svm import PegasosSVM


def test_to_pm1():
    out = PegasosSVM._to_pm1(np.array([0, 1, 1, 0]), np.array([0, 1]))
    assert out.tolist() == [-1.0, 1.0, 1.0, -1.0]


def test_fit_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    m = PegasosSVM(lambda_=0.1, n_iters=200, random_state=0).fit(X, y)
    assert m.w.shape == (2,)


if __name__ == "__main__":
    test_to_pm1()
    test_fit_runs()
    print("OK")