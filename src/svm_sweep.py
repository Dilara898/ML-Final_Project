import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_prep import prepare_data
from .svm import PegasosSVM

LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_ITERS = 50_000
SEED = 42


def accuracy(model, X, y):
    return float((model.predict(X) == np.asarray(y)).mean())


def main():
    d = prepare_data()
    X_tr, y_tr = d["X_train_scaled"], np.asarray(d["tier_train"])
    X_val, y_val = d["X_val_scaled"], np.asarray(d["tier_val"])
    print("train:", X_tr.shape, "val:", X_val.shape)
    print(f"epoch sayı ≈ {N_ITERS / len(X_tr):.2f}")

    rows = []
    plt.figure(figsize=(7, 4))
    for lam in LAMBDAS:
        t0 = time.time()
        m = PegasosSVM(lambda_=lam, n_iters=N_ITERS, random_state=SEED).fit(X_tr, y_tr)
        secs = time.time() - t0
        sv_share = m.support_vector_mask(X_tr, y_tr).mean()
        rows.append((lam, np.linalg.norm(m.w), m.margin_width(), sv_share,
                     accuracy(m, X_tr, y_tr), accuracy(m, X_val, y_val), secs))
        its, js = zip(*m.history_)
        plt.plot(its, js, label=f"lambda={lam:g}")

    print(f"{'lambda':>8} {'||w||':>9} {'marja':>9} {'SV payi':>8} "
          f"{'train':>7} {'val':>7} {'san':>6}")
    for lam, nw, mw, sv, a_tr, a_val, secs in rows:
        print(f"{lam:>8g} {nw:>9.3f} {mw:>9.3f} {sv:>8.3f} "
              f"{a_tr:>7.3f} {a_val:>7.3f} {secs:>6.1f}")

    plt.yscale("log")
    plt.xlabel("iterasiya")
    plt.ylabel("J (obyektiv)")
    plt.title("Pegasos obyektivinin konvergensiyasi")
    plt.legend()
    plt.tight_layout()
    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/svm_lambda_convergence.png", dpi=150)


if __name__ == "__main__":
    main()