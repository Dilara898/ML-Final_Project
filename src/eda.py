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

def plot_rooms(training_df):
    """Show observed residential room counts and median prices."""
    category_labels = {
        "Yeni tikili": "New-build apartments",
        "Köhnə tikili": "Older-build apartments",
        "Həyət evi/Bağ evi": "Houses / garden houses",
    }

    residential = training_df.loc[
        training_df["property_category"].isin(category_labels)
    ].copy()

    missing_count = residential["rooms"].isna().sum()
    observed = residential.dropna(subset=["rooms"]).copy()

    # Group rare large room counts for readable plots.
    observed["room_group"] = (
        observed["rooms"]
        .astype(int)
        .map(lambda value: str(value) if value < 7 else "7+")
    )

    room_order = ["1", "2", "3", "4", "5", "6", "7+"]

    counts = (
        observed["room_group"]
        .value_counts()
        .reindex(room_order, fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(
        room_order,
        counts.to_numpy(),
        color="steelblue",
    )
    axes[0].bar_label(
        bars,
        labels=[f"{value:,}" for value in counts],
        padding=3,
        fontsize=9,
    )
    axes[0].set_title("Observed residential room counts")
    axes[0].set_xlabel("Number of rooms")
    axes[0].set_ylabel("Number of listings")
    axes[0].set_ylim(0, counts.max() * 1.15)

    for category, label in category_labels.items():
        subset = observed.loc[
            observed["property_category"].eq(category)
        ]

        medians = (
            subset.groupby("room_group")["price"]
            .median()
            .reindex(room_order)
        )

        axes[1].plot(
            room_order,
            medians / 1_000,
            marker="o",
            label=label,
        )

    axes[1].set_title("Median price by rooms and property category")
    axes[1].set_xlabel("Number of rooms")
    axes[1].set_ylabel("Median price (thousand AZN)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Training residential listings — "
        f"{missing_count:,} missing room counts excluded from these plots"
    )
    fig.text(
    0.5,
    0.01,
    "Older-build apartments: 6 rooms n=25; 7+ rooms n=9. "
    "Interpret small-group medians cautiously.",
    ha="center",
    fontsize=9,
)

    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "eda_rooms.png"

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("Saved:", output_path)

    print("\nObserved room-group counts by property category:")
    print(
        observed.groupby(["property_category", "room_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=room_order, fill_value=0)
        .to_string()
    )

def plot_geographic_prices(training_df):
    """Plot observed training coordinates coloured by sale price."""
    from matplotlib.colors import LogNorm

    observed = training_df.dropna(subset=["lat", "lng"]).copy()
    missing_count = len(training_df) - len(observed)

    # Shuffle drawing order so expensive listings aren't always on top.
    observed = observed.sample(frac=1, random_state=42)

    baku = observed.loc[observed["city"].eq("bakı")]

    # Both panels use the same colour scale.
    colour_scale = LogNorm(
        vmin=observed["price"].min(),
        vmax=observed["price"].max(),
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 6),
        constrained_layout=True,
    )

    panels = [
        (axes[0], observed, "All training locations"),
        (axes[1], baku, "Listings labelled Bakı"),
    ]

    for ax, subset, title in panels:
        points = ax.scatter(
            subset["lng"],
            subset["lat"],
            c=subset["price"],
            norm=colour_scale,
            cmap="viridis",
            s=12,
            alpha=0.7,
            edgecolors="none",
        )

        ax.set_title(f"{title} (n={len(subset):,})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        ax.ticklabel_format(
            axis="both",
            style="plain",
            useOffset=False,
        )

        # Approximate geographic proportions at the panel's latitude.
        if len(subset) > 0:
            mean_latitude = subset["lat"].mean()
            ax.set_aspect(
                1 / np.cos(np.deg2rad(mean_latitude))
            )

    colour_bar = fig.colorbar(
        points,
        ax=axes,
        shrink=0.75,
        pad=0.03,
    )
    colour_bar.set_label("Sale price (AZN; logarithmic colour scale)")

    fig.suptitle(
        "Training property locations and prices\n"
        f"{missing_count:,} missing coordinate pairs excluded; "
        "no background map"
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "eda_geographic_prices.png"

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("Saved:", output_path)



def plot_class_balance(data):
    """Show class counts and percentages using the training-only threshold."""
    splits = [
        ("Training", data["tier_train"]),
        ("Validation", data["tier_val"]),
        ("Test", data["tier_test"]),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(12, 5),
        sharey=True,
    )

    for ax, (name, labels) in zip(axes, splits):
        counts = np.bincount(
            np.asarray(labels, dtype=int),
            minlength=2,
        )
        percentages = counts / counts.sum() * 100

        bars = ax.bar(
            ["Standard", "Premium"],
            percentages,
            color=["steelblue", "darkorange"],
        )

        ax.bar_label(
            bars,
            labels=[
                f"{percentage:.2f}%\n(n={count:,})"
                for percentage, count in zip(percentages, counts)
            ],
            padding=4,
            fontsize=10,
        )

        ax.set_title(f"{name} (n={len(labels):,})")
        ax.set_ylim(0, 65)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)

        print(
            f"{name}: standard={counts[0]:,}, "
            f"premium={counts[1]:,}"
        )

    axes[0].set_ylabel("Percentage of listings")

    threshold = data["tier_threshold"]
    fig.suptitle(
        f"Price-tier balance — training median: {threshold:,.0f} AZN\n"
        "Standard: price ≤ threshold | Premium: price > threshold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "eda_class_balance.png"

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("Saved:", output_path)





if __name__ == "__main__":
    data = prepare_data()
    training_df = data["training_df"]

    plot_price_distributions(training_df)
    plot_area_vs_price(training_df)
    plot_rooms(training_df)
    plot_geographic_prices(training_df)
    plot_class_balance(data)
    