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

def plot_area_vs_price(training_df):
    """Plot original cleaned area versus price by property category."""
    category_labels = {
        "Yeni tikili": "New-build apartments",
        "Köhnə tikili": "Older-build apartments",
        "Həyət evi/Bağ evi": "Houses / garden houses",
        "Torpaq": "Land",
        "Obyekt": "Commercial properties",
        "Ofis": "Offices",
        "Qaraj": "Garages",
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for ax, (category, label) in zip(axes, category_labels.items()):
        subset = training_df.loc[
            training_df["property_category"].eq(category)
        ]

        ax.scatter(
            subset["area_m2"],
            subset["price"],
            s=10,
            alpha=0.25,
            color="steelblue",
            edgecolors="none",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{label} (n={len(subset):,})")
        ax.set_xlabel("Area (m²; log scale)")
        ax.set_ylabel("Price (AZN; log scale)")
        ax.grid(True, alpha=0.2)

    axes[-1].axis("off")

    fig.suptitle(
        "Training listings: area versus sale price\n"
        "Before area capping; axis ranges differ across panels",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "eda_area_vs_price.png"

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("Saved:", output_path)





if __name__ == "__main__":
    data = prepare_data()
    training_df = data["training_df"]

    plot_price_distributions(training_df)
    plot_area_vs_price(training_df)