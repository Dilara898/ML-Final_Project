import numpy as np
from src.svm import PegasosSVM, RandomFourierFeatures


def test_to_pm1():
    out = PegasosSVM._to_pm1(np.array([0, 1, 1, 0]), np.array([0, 1]))
    assert out.tolist() == [-1.0, 1.0, 1.0, -1.0]


def test_fit_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    m = PegasosSVM(lambda_=0.1, n_iters=200, random_state=0).fit(X, y)
    assert m.w.shape == (2,)

def test_objective_decreases():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    m = PegasosSVM(lambda_=0.01, n_iters=5000, random_state=42).fit(X, y)
    assert len(m.history_) > 0
    assert m.history_[-1][1] < 1.0

def test_margin_and_support_vectors():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    m = PegasosSVM(lambda_=0.01, n_iters=5000, random_state=42).fit(X, y)
    assert m.margin_width() > 0
    mask = m.support_vector_mask(X, y)
    assert mask.dtype == bool
    assert 0 < mask.sum() < len(y)

def test_vs_sklearn():
    from sklearn.svm import SVC

    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-1.5, 1, (100, 2)), rng.normal(1.5, 1, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    lam = 0.01
    ours = PegasosSVM(lambda_=lam, n_iters=20000, random_state=42).fit(X, y)
    C = 1.0 / (len(y) * lam)
    ref = SVC(kernel="linear", C=C).fit(X, y)

    agree = (ours.predict(X) == ref.predict(X)).mean()
    w_ref = ref.coef_[0]
    cos = ours.w @ w_ref / (np.linalg.norm(ours.w) * np.linalg.norm(w_ref))
    print(f"agreement={agree:.3f}, cosine(w)={cos:.3f}")
    assert agree > 0.95, agree
def test_rff_nonlinear():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1.5, size=(400, 2))
    y = (np.linalg.norm(X, axis=1) < 1.5).astype(int)

    lin = PegasosSVM(lambda_=0.01, n_iters=10000, random_state=42).fit(X, y)
    acc_lin = (lin.predict(X) == y).mean()

    rff = RandomFourierFeatures(n_components=200, gamma=0.5, random_state=42).fit(X)
    Z = rff.transform(X)
    m = PegasosSVM(lambda_=0.001, n_iters=20000, random_state=42).fit(Z, y)
    acc_rff = (m.predict(Z) == y).mean()

    print(f"linear={acc_lin:.3f}, rff={acc_rff:.3f}")
    assert acc_rff > 0.9 and acc_rff > acc_lin + 0.1
if __name__ == "__main__":
    test_to_pm1()
    test_fit_runs()
    test_objective_decreases()
    test_margin_and_support_vectors()
    test_vs_sklearn()
    test_rff_nonlinear()
    print("OK")