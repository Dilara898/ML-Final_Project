from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_prep import prepare_data


FIGURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "report"
    / "figures"
)


def plot_price_distributions(training_df):
    """Plot training sale prices and their natural logarithms."""
    prices = training_df["price"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(
        prices / 1_000_000,
        bins=60,
        color="steelblue",
        edgecolor="white",
    )
    axes[0].set_title("Training sale prices")
    axes[0].set_xlabel("Price (million AZN)")
    axes[0].set_ylabel("Number of listings")
    axes[0].ticklabel_format(
        axis="x",
        style="plain",
        useOffset=False,
    )

    axes[1].hist(
        np.log(prices),
        bins=60,
        color="darkorange",
        edgecolor="white",
    )
    axes[1].set_title("Natural logarithm of training prices")
    axes[1].set_xlabel("ln(Price / 1 AZN)")
    axes[1].set_ylabel("Number of listings")

    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "eda_price_distributions.png"

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("Saved:", output_path)


if __name__ == "__main__":
    data = prepare_data()
    plot_price_distributions(data["training_df"])