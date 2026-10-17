import time
import numpy as np

from .data_prep import prepare_data
from .svm import PegasosSVM, RandomFourierFeatures

SEED = 42
ITERS = [50_000, 200_000, 500_000]


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


def main():
    d = prepare_data()
    X_tr, y_tr = d["X_train_scaled"], np.asarray(d["tier_train"])
    X_val, y_val = d["X_val_scaled"], np.asarray(d["tier_val"])
    n = len(X_tr)

    rows = []  # (config, n_iters, val_acc, seconds)

    for lam in [0.003, 0.01, 0.03]:
        for n_iters in ITERS:
            t0 = time.time()
            m = PegasosSVM(lambda_=lam, n_iters=n_iters, random_state=SEED).fit(X_tr, y_tr)
            rows.append((f"linear lam={lam:g}", n_iters,
                         accuracy(m, X_val, y_val), time.time() - t0))

    gamma, lam = 0.003, 1e-4
    for comps in [300, 1000]:
        rff = RandomFourierFeatures(
            n_components=comps, gamma=gamma, random_state=SEED
        ).fit(X_tr)
        Z_tr, Z_val = rff.transform(X_tr), rff.transform(X_val)
        for n_iters in ITERS:
            t0 = time.time()
            m = PegasosSVM(lambda_=lam, n_iters=n_iters, random_state=SEED).fit(Z_tr, y_tr)
            rows.append((f"rff g={gamma:g} lam={lam:g} D={comps}", n_iters,
                         accuracy(m, Z_val, y_val), time.time() - t0))

    print(f"{'konfiqurasiya':<30} {'n_iters':>8} {'epoch':>6} {'val':>7} {'san':>6}")
    for cfg, n_iters, a_val, secs in rows:
        print(f"{cfg:<30} {n_iters:>8} {n_iters / n:>6.1f} {a_val:>7.3f} {secs:>6.1f}")

    best = max(rows, key=lambda r: r[2])
    print(f"ən yaxşı (validation-a görə): {best[0]}, n_iters={best[1]}, val={best[2]:.3f}")


if __name__ == "__main__":
    main()