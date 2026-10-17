import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_prep import prepare_data
from .svm import PegasosSVM, RandomFourierFeatures

GAMMAS = [0.0005, 0.001, 0.003, 0.01, 0.03]
LAMBDAS = [1e-5, 1e-4, 1e-3]
N_COMPONENTS = 300
N_ITERS = 50_000
SEED = 42


def accuracy(model, X, y):
    return float((model.predict(X) == np.asarray(y)).mean())


def main():
    d = prepare_data()
    X_tr, y_tr = d["X_train_scaled"], np.asarray(d["tier_train"])
    X_val, y_val = d["X_val_scaled"], np.asarray(d["tier_val"])

    results = []
    for gamma in GAMMAS:
        rff = RandomFourierFeatures(
            n_components=N_COMPONENTS, gamma=gamma, random_state=SEED
        ).fit(X_tr)
        Z_tr, Z_val = rff.transform(X_tr), rff.transform(X_val)
        for lam in LAMBDAS:
            t0 = time.time()
            m = PegasosSVM(lambda_=lam, n_iters=N_ITERS, random_state=SEED).fit(Z_tr, y_tr)
            secs = time.time() - t0
            results.append((gamma, lam, accuracy(m, Z_tr, y_tr),
                            accuracy(m, Z_val, y_val), secs))

    print(f"{'gamma':>8} {'lambda':>8} {'train':>7} {'val':>7} {'san':>6}")
    for gamma, lam, a_tr, a_val, secs in results:
        print(f"{gamma:>8g} {lam:>8g} {a_tr:>7.3f} {a_val:>7.3f} {secs:>6.1f}")

    best = max(results, key=lambda r: r[3])
    print(f"ən yaxşı (validation-a görə): gamma={best[0]:g}, "
          f"lambda={best[1]:g}, val={best[3]:.3f}")

    plt.figure(figsize=(7, 4))
    for lam in LAMBDAS:
        rows = [r for r in results if r[1] == lam]
        plt.plot([r[0] for r in rows], [r[3] for r in rows],
                 marker="o", label=f"lambda={lam:g}")
    plt.xscale("log")
    plt.xlabel("gamma")
    plt.ylabel("validation dəqiqliyi")
    plt.title("RFF + Pegasos: gamma və lambda təsiri")
    plt.legend()
    plt.tight_layout()
    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/svm_rff_gamma.png", dpi=150)


if __name__ == "__main__":
    main()