import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_prep import prepare_data
from .svm import PegasosSVM

LAMBDA = 3e-3
N_ITERS = 500_000
SEED = 42
CONFIGS = [("pegasos", None), ("constant", 0.001), ("constant", 0.01), ("constant", 0.1)]


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


def main():
    d = prepare_data()
    X_tr, y_tr = d["X_train_scaled"], np.asarray(d["tier_train"])
    X_val, y_val = d["X_val_scaled"], np.asarray(d["tier_val"])

    print(f"{'schedule':<10}{'eta0':>7}{'||w||':>9}{'marja':>8}{'SV':>7}"
          f"{'train':>7}{'val':>7}{'J_son':>9}{'san':>6}")
    plt.figure(figsize=(7, 4))
    for sched, eta0 in CONFIGS:
        t0 = time.time()
        m = PegasosSVM(lambda_=LAMBDA, n_iters=N_ITERS, random_state=SEED,
                       lr_schedule=sched, eta0=eta0 if eta0 else 0.1).fit(X_tr, y_tr)
        secs = time.time() - t0
        label = "pegasos" if sched == "pegasos" else f"constant eta0={eta0:g}"
        print(f"{sched:<10}{(eta0 if eta0 else float('nan')):>7g}"
              f"{np.linalg.norm(m.w):>9.3f}{m.margin_width():>8.3f}"
              f"{m.support_vector_mask(X_tr, y_tr).mean():>7.3f}"
              f"{accuracy(m, X_tr, y_tr):>7.3f}{accuracy(m, X_val, y_val):>7.3f}"
              f"{m.history_[-1][1]:>9.4f}{secs:>6.1f}")
        its, js = zip(*m.history_)
        plt.plot(its, js, label=label)

    plt.yscale("log")
    plt.xlabel("iterasiya")
    plt.ylabel("J (obyektiv)")
    plt.title("Öyrənmə sürəti cədvəlinin təsiri (lambda=3e-3)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/svm_lr_schedule.png", dpi=150)


if __name__ == "__main__":
    main()