"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""
# Price audit:
# All 100,775 rows have positive, non-missing prices and AZN currency.
# price and total_price differ in 13 rows with different scrape timestamps.
# Keep the specified target, price; do not overwrite it.
# Exclude total_price and unit_price from model inputs.

# Billing-cycle audit:
# Rental keywords also appear in sale listings describing rental income
# or parking fees, so keyword matches are not automatically removed.
# No daily/monthly conversion is applied to the sale-price target.
# Reviewed a sample of flagged descriptions; some targets remain ambiguous.
# This check is not an exhaustive verification of all listings.


from __future__ import annotations

import numpy as np
import pandas as pd
# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"   # change if your filename differs
COLUMN_TRANSLATIONS = {
    "Binanın növü": "building_type",
    "Kateqoriya": "property_category",
    "Mərtəbə": "floor_raw",
    "Otaq sayı": "rooms",
    "Sahə": "area_raw",
    "Torpaq sahəsi": "land_area_raw",
    "Təmir": "renovation",
    "Çıxarış": "ownership_document",
    "İpoteka": "mortgage_available",
}

# Columns that LEAK the target (derived from price). Drop these before training.
# TODO: list the actual leakage columns you find (e.g. a unit/per-m2 price).
LEAKAGE_COLUMNS: list[str] = [
    # "unit_price",
    # "total_price",
]

BILLING_REVIEW_IDS = {
    "189e08a6-095e-49c1-97a6-1d2fd90578d2",
    "92985fed-de67-4547-838e-2b551388a4ed",
}
def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    df = pd.read_csv(path, encoding="utf-8")
    return df
def translate_columns(df):
    """Return a DataFrame with translated column names."""
    return df.rename(columns=COLUMN_TRANSLATIONS)

def parse_floors(df):
    """Extract floor number and total floors from floor_raw."""
    df = df.copy()

    parts = df["floor_raw"].str.extract(
        r"^\s*(\d+)\s*/\s*(\d+)\s*$"
    )

    df["floor_number"] = pd.to_numeric(parts[0], errors="coerce")
    df["total_floors"] = pd.to_numeric(parts[1], errors="coerce")

    return df
def parse_area(df):
    """Convert area_raw into numeric square metres."""
    df = df.copy()

    text = (
        df["area_raw"]
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
    )

    parts = text.str.extract(r"^(\d+(?:\.\d+)?)(m²|sot)$")

    numbers = pd.to_numeric(parts[0], errors="coerce")
    multipliers = parts[1].map({"m²": 1, "sot": 100})

    df["area_m2"] = numbers * multipliers

    return df

def parse_land_area(df):
    """Convert land_area_raw from sot to square metres."""
    # All non-missing land areas parsed successfully from sot to m².
    # Very large plots and repeated 0.1-sot values need later investigation.
    # No values were corrected or removed based on size alone.
    df = df.copy()

    text = (
        df["land_area_raw"]
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
    )

    numbers = text.str.extract(
        r"^(\d+(?:\.\d+)?)sot$",
        expand=False,
    )

    df["land_area_m2"] = pd.to_numeric(
        numbers, errors="coerce"
    ) * 100

    return df
def prepare_missing_values(df):
    """Preserve missingness and label unknown categorical values."""
    df = df.copy()

    numeric_columns = [
        "land_area_m2",
        "floor_number",
        "total_floors",
        "rooms",
    ]

    for column in numeric_columns:
        df[f"{column}_missing"] = df[column].isna().astype(int)

    categorical_columns = [
        "renovation",
        "mortgage_available",
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna("unknown")

    return df

def clean(df):
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    raise NotImplementedError("TODO: clean the data")


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    raise NotImplementedError("TODO: build X, y")


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    raise NotImplementedError("TODO: derive the price-tier label")


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    raise NotImplementedError("TODO: split the data")


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    raise NotImplementedError("TODO: standardize using train stats")

if __name__ == "__main__":
    raw_df = load_raw()
    df = translate_columns(raw_df)
    df = parse_floors(df)
    df = parse_area(df)
    df = parse_land_area(df)
    df = prepare_missing_values(df)

    # print("\nArea parsing examples:")
    # print(
    #     df[["area_raw", "area_m2"]]
    #     .head(10)
    #     .to_string(index=False)
    # )

    # failed_area = df["area_raw"].notna() & df["area_m2"].isna()

    # print("\nArea parsing failures:", failed_area.sum())
    # print(df.loc[failed_area, "area_raw"].value_counts().head(20))

    # invalid_area = df["area_m2"] <= 0

    # print("\nZero or negative areas:", invalid_area.sum())
    # print(
    #     df.loc[invalid_area, ["area_raw", "area_m2"]]
    #     .head(20)
    #     .to_string(index=False)
    # )

    # print("\nArea summary in square metres:")
    # print(df["area_m2"].describe())
    # print("Floor parsing examples:")
    # print(
    #     df[
    #         ["floor_raw", "floor_number", "total_floors"]
    #     ].head(15).to_string(index=False)
    # )

    # failed = df["floor_raw"].notna() & df["floor_number"].isna()

    # print("\nNon-missing values we could not parse:", failed.sum())
    # print(df.loc[failed, "floor_raw"].value_counts().head(20))

    # suspicious_floors = (
    #     (df["floor_number"] > df["total_floors"])
    #     | (df["total_floors"] <= 0)
    #     | (df["floor_number"] <= 0)
    # )

    # print("\nSuspicious floor rows:", suspicious_floors.sum())

    # print(
    #     df.loc[
    #         suspicious_floors,
    #         ["property_category", "floor_raw",
    #          "floor_number", "total_floors"]
    #     ].to_string(index=False)
    # )
    # area_units = (
    #     df["area_raw"]
    #     .str.replace(r"[\d\s.,]+", "", regex=True)
    #     .str.lower()
    # )

    # print("\nArea units:")
    # print(area_units.value_counts(dropna=False))


    # inspection_columns = [
    #     "property_category",
    #     "area_raw",
    #     "area_m2",
    #     "price",
    # ]

    # print("\n10 largest areas:")
    # print(
    #     df.nlargest(10, "area_m2")[inspection_columns]
    #     .to_string(index=False)
    # )

    # print("\n10 smallest areas:")
    # print(
    #     df.nsmallest(10, "area_m2")[inspection_columns]
    #     .to_string(index=False)
    # )
    


    # land_units = (
    #     df["land_area_raw"]
    #     .str.replace(r"[\d\s.,]+", "", regex=True)
    #     .str.lower()
    # )

    # print("\nLand-area units:")
    # print(land_units.value_counts(dropna=False))

    # print("\nLand-area parsing examples:")
    # print(
    #     df.loc[
    #         df["land_area_raw"].notna(),
    #         ["land_area_raw", "land_area_m2"]
    #     ].head(10).to_string(index=False)
    # )

    # failed_land = (
    #     df["land_area_raw"].notna()
    #     & df["land_area_m2"].isna()
    # )

    # print("\nLand-area parsing failures:", failed_land.sum())
    # print(
    #     df.loc[failed_land, "land_area_raw"]
    #     .value_counts()
    #     .head(20)
    # )

    # invalid_land = df["land_area_m2"] <= 0

    # print("\nZero or negative land areas:", invalid_land.sum())
    # print(
    #     df.loc[
    #         invalid_land,
    #         ["land_area_raw", "land_area_m2"]
    #     ].head(20).to_string(index=False)
    # )

    # print("\nMissing land areas:", df["land_area_m2"].isna().sum())

    # print("\nLand-area summary:")
    # print(df["land_area_m2"].describe())

    # land_columns = [
    #     "property_category",
    #     "area_raw",
    #     "land_area_raw",
    #     "land_area_m2",
    #     "price",
    # ]

    # available_land = df.loc[df["land_area_m2"].notna()]

    # print("\n10 largest land areas:")
    # print(
    #     available_land.nlargest(10, "land_area_m2")[land_columns]
    #     .to_string(index=False)
    # )

    # print("\n10 smallest land areas:")
    # print(
    #     available_land.nsmallest(10, "land_area_m2")[land_columns]
    #     .to_string(index=False)
    # )
    # print("\nCurrency combinations:")
    # print(
    #     df[["currency_x", "currency_y"]]
    #     .value_counts(dropna=False)
    # )

    # print("\nMissing prices:", df["price"].isna().sum())
    # print("Zero or negative prices:", (df["price"] <= 0).sum())

    # same_price = df["price"].eq(df["total_price"])

    # print("\nRows where price equals total_price:", same_price.sum())
    # print("Rows where they differ:", (~same_price).sum())

    # print("\nListing-text examples:")
    # for column in ["products_label", "attributes"]:
    #     print(f"\n--- {column} ---")
    #     print(
    #         df[column]
    #         .dropna()
    #         .drop_duplicates()
    #         .head(10)
    #         .to_string(index=False)
    #     )

    # different_prices = df["price"].ne(df["total_price"])

    # print("\nListings with different prices:")
    # print(
    #     df.loc[
    #         different_prices,
    #         [
    #             "estate_id",
    #             "property_category",
    #             "price",
    #             "total_price",
    #             "unit_price",
    #             "area_raw",
    #             "datetime_scrape_x",
    #             "datetime_scrape_y",
    #         ],
    #     ].to_string(index=False)
    # )
    # print("\nExample listing URLs:")
    # print(df["rel_url"].dropna().head(10).to_string(index=False))

    # rental_pattern = (
    #     r"kiray|icar|rent|günlük|gunluk|gündəlik|gundelik|"
    #     r"aylıq|ayliq|daily|monthly"
    # )

    # rental_mentions = (
    #     df["description"]
    #     .fillna("")
    #     .str.contains(rental_pattern, case=False, regex=True)
    # )

    # print("\nDescriptions mentioning rental terms:", rental_mentions.sum())

    # examples = df.loc[
    #     rental_mentions,
    #     ["estate_id", "price", "description"]
    # ].head(10)

    # for _, row in examples.iterrows():
    #     print("\nListing:", row["estate_id"])
    #     print("Price:", row["price"])
    #     print("Description:", row["description"])


    # feature_candidates = [
    #     "property_category",
    #     "building_type",
    #     "rooms",
    #     "area_m2",
    #     "land_area_m2",
    #     "floor_number",
    #     "total_floors",
    #     "renovation",
    #     "ownership_document",
    #     "mortgage_available",
    #     "city",
    #     "location",
    #     "lat",
    #     "lng",
    # ]

    # missing_summary = pd.DataFrame({
    #     "missing_count": df[feature_candidates].isna().sum(),
    #     "missing_percent": (
    #         df[feature_candidates].isna().mean() * 100
    #     ).round(2),
    # })

    # print("\nMissing values in candidate features:")
    # print(
    #     missing_summary
    #     .sort_values("missing_percent", ascending=False)
    #     .to_string()
    # )

    # columns_to_check = [
    #     "land_area_m2",
    #     "floor_number",
    #     "total_floors",
    #     "rooms",
    # ]

    # missing_by_category = (
    #     df[columns_to_check]
    #     .isna()
    #     .groupby(df["property_category"])
    #     .mean()
    #     .mul(100)
    #     .round(1)
    # )

    # print("\nNumber of rows per property category:")
    # print(df["property_category"].value_counts())

    # print("\nMissing percentage within each category:")
    # print(missing_by_category.to_string())

    print("\nMissingness indicators:")
    print(
        df[
            [
                "land_area_m2_missing",
                "floor_number_missing",
                "total_floors_missing",
                "rooms_missing",
            ]
        ].sum()
    )

    print("\nRenovation categories:")
    print(df["renovation"].value_counts(dropna=False))

    print("\nMortgage categories:")
    print(df["mortgage_available"].value_counts(dropna=False))