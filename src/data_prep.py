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
LEAKAGE_COLUMNS = [
    "unit_price",
    "total_price",
]

BILLING_REVIEW_IDS = {
    "189e08a6-095e-49c1-97a6-1d2fd90578d2",
    "92985fed-de67-4547-838e-2b551388a4ed",
}
AMBIGUOUS_PRICE_LISTINGS = {
    "/items/4656308": (
        "Rental offer with equipment; listed price is 23000 AZN "
        "but stated rent is 2000 AZN. Sale-price meaning is unclear."
    ),
    "/items/4604180": (
        "Description offers the property for rent but also addresses "
        "buyers. The meaning of the listed 510000 AZN is unclear."
    ),
}

NUMERIC_FEATURES = [
    "area_m2",
    "land_area_m2",
    "rooms",
    "floor_number",
    "total_floors",
    "lat",
    "lng",
]

CATEGORICAL_FEATURES = [
    "property_category",
    "renovation",
    "ownership_document",
    "mortgage_available",
    "city",
    "location",
]

MISSING_INDICATORS = [
    "land_area_m2_missing",
    "floor_number_missing",
    "total_floors_missing",
    "rooms_missing",
    "coordinates_missing",
]




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


def remove_duplicate_listings(df):
    """Keep the latest collected row for each listing URL."""
    df = df.copy()

    df["listing_url"] = (
        df["estate_rel_url_x"].astype("string").str.strip()
    )

    # Treat timestamps without timezone information as UTC.
    df["_scrape_time"] = pd.to_datetime(
        df["datetime_scrape_x"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    if df["listing_url"].isna().any() or df["listing_url"].eq("").any():
        raise ValueError("Missing listing URLs: review before deduplication.")

    if df["_scrape_time"].isna().any():
        raise ValueError("Invalid scrape timestamps: review before deduplication.")

    df = df.sort_values("_scrape_time", kind="stable")

    df = df.drop_duplicates(
        subset="listing_url",
        keep="last",
    )

    return df.drop(columns="_scrape_time").reset_index(drop=True)

def drop_leakage_columns(df):
    """Remove price-derived columns from the working table."""
    return df.drop(columns=LEAKAGE_COLUMNS)

def select_features(df):
    """Separate selected input features from the price target."""
    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + MISSING_INDICATORS
    )

    X = df[feature_columns].copy()
    y = df["price"].copy()

    return X, y

def clean_coordinates(df):
    """Mark invalid or manually identified unreliable coordinates as missing."""
    df = df.copy()

    invalid = (
        ~df["lat"].between(-90, 90)
        | ~df["lng"].between(-180, 180)
        | df["lat"].eq(0)
        | df["lng"].eq(0)
    )

    # Reviewed listing: coordinates conflict with its stated Baku location.
    known_bad = df["listing_url"].eq("/items/4596070")

    unreliable = invalid | known_bad

    df.loc[unreliable, ["lat", "lng"]] = np.nan
    df["coordinates_missing"] = unreliable.astype(int)

    return df

def exclude_ambiguous_prices(df):
    """Exclude manually reviewed listings with unclear sale-price targets."""
    excluded = df["listing_url"].isin(AMBIGUOUS_PRICE_LISTINGS)

    return df.loc[~excluded].copy().reset_index(drop=True)


def clean(df):
    """Apply fixed cleaning rules without learning training statistics."""
    df = translate_columns(df)
    df = parse_floors(df)
    df = parse_area(df)
    df = parse_land_area(df)
    df = prepare_missing_values(df)
    df = remove_duplicate_listings(df)
    df = exclude_ambiguous_prices(df)
    df = clean_coordinates(df)
    df = drop_leakage_columns(df)

    return df


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    raise NotImplementedError("TODO: build X, y")


def make_tier_label(y_price, threshold=None):
    """Create price tiers; learn the threshold only from training prices."""
    prices = np.asarray(y_price, dtype=float)

    if prices.size == 0 or not np.isfinite(prices).all():
        raise ValueError("Prices must be non-empty and finite.")

    if threshold is None:
        threshold = float(np.median(prices))

    y_tier = (prices > threshold).astype(int)

    return y_tier, threshold


def train_val_test_split(
    X, y, val_size=0.15, test_size=0.15, seed=SEED
):
    """Select training first, then stratify validation/test by price tier."""
    if not (0 < val_size < 1 and 0 < test_size < 1):
        raise ValueError("Validation and test fractions must be between 0 and 1.")

    if val_size + test_size >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have matching rows and indices.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_train = int(len(X) * (1 - val_size - test_size))
    train_indices = indices[:n_train]
    remaining_indices = indices[n_train:]

    _, threshold = make_tier_label(y.iloc[train_indices])

    remaining_tiers, _ = make_tier_label(
        y.iloc[remaining_indices], threshold=threshold
    )

    validation_parts = []
    test_parts = []

    validation_fraction = val_size / (val_size + test_size)

    for label in [0, 1]:
        class_indices = remaining_indices[remaining_tiers == label]
        class_indices = rng.permutation(class_indices)

        n_validation = int(len(class_indices) * validation_fraction)

        validation_parts.append(class_indices[:n_validation])
        test_parts.append(class_indices[n_validation:])

    val_indices = rng.permutation(np.concatenate(validation_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))

    return (
        X.iloc[train_indices].copy(),
        y.iloc[train_indices].copy(),
        X.iloc[val_indices].copy(),
        y.iloc[val_indices].copy(),
        X.iloc[test_indices].copy(),
        y.iloc[test_indices].copy(),
    )

def fit_area_caps(X_train, quantile=0.99, min_count=100):
    """Learn upper area limits from training data by property category."""
    caps = {}

    for column in ["area_m2", "land_area_m2"]:
        grouped = X_train.groupby("property_category")[column]

        counts = grouped.count()
        limits = grouped.quantile(quantile)

        caps[column] = limits[counts >= min_count].to_dict()

    return caps


def apply_area_caps(X, caps):
    """Apply learned area limits without changing the input table."""
    result = X.copy()

    for column, category_limits in caps.items():
        limits = result["property_category"].map(category_limits)

        above_limit = (
            result[column].notna()
            & limits.notna()
            & result[column].gt(limits)
        )

        result.loc[above_limit, column] = limits.loc[above_limit]

    return result

def fit_numeric_imputer(X_train):
    """Learn numerical replacement values from training data only."""
    category_medians = (
        X_train.groupby("property_category")[NUMERIC_FEATURES]
        .median()
    )

    overall_medians = X_train[NUMERIC_FEATURES].median()

    if overall_medians.isna().any():
        raise ValueError("A numerical feature has no observed training values.")

    return {
        "category_medians": category_medians,
        "overall_medians": overall_medians,
    }


def apply_numeric_imputer(X, imputer):
    """Fill numerical gaps using previously learned medians."""
    result = X.copy()

    for column in NUMERIC_FEATURES:
        category_values = result["property_category"].map(
            imputer["category_medians"][column]
        )

        result[column] = (
            result[column]
            .fillna(category_values)
            .fillna(imputer["overall_medians"][column])
        )

    return result

def fit_category_encoder(X_train):
    """Learn categorical values from training data only."""
    categories = {}

    for column in CATEGORICAL_FEATURES:
        values = (
            X_train[column]
            .astype("string")
            .fillna("unknown")
        )

        categories[column] = sorted(values.unique().tolist())

    return categories


def apply_category_encoder(X, categories):
    """Create identical numerical feature columns for every split."""
    result = X[NUMERIC_FEATURES + MISSING_INDICATORS].astype(float).copy()
    encoded_columns = {}

    for column in CATEGORICAL_FEATURES:
        values = X[column].astype("string").fillna("unknown")

        for category in categories[column]:
            name = f"{column}={category}"
            encoded_columns[name] = values.eq(category).astype(float)

    encoded = pd.DataFrame(encoded_columns, index=X.index)

    return pd.concat([result, encoded], axis=1)

def standardize(X_tr, *others):
    """Scale all splits using training means and standard deviations."""
    train = np.asarray(X_tr, dtype=float)

    if train.ndim != 2 or not np.isfinite(train).all():
        raise ValueError("Training features must be a finite 2D array.")

    means = train.mean(axis=0)
    stds = train.std(axis=0)

    # Avoid division by zero for columns constant in training.
    safe_stds = np.where(stds == 0, 1.0, stds)

    scaled = [(train - means) / safe_stds]

    for X in others:
        values = np.asarray(X, dtype=float)

        if values.ndim != 2 or values.shape[1] != train.shape[1]:
            raise ValueError("All splits must have the same feature count.")

        if not np.isfinite(values).all():
            raise ValueError("Features must contain only finite values.")

        scaled.append((values - means) / safe_stds)

    return tuple(scaled)

if __name__ == "__main__":
    raw_df = load_raw()
    df = clean(raw_df)
    X, y = select_features(df)

    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)

    tier_tr, threshold = make_tier_label(y_tr)
    tier_val, _ = make_tier_label(y_val, threshold=threshold)
    tier_te, _ = make_tier_label(y_te, threshold=threshold)


    area_caps = fit_area_caps(X_tr)

    X_tr_capped = apply_area_caps(X_tr, area_caps)
    X_val_capped = apply_area_caps(X_val, area_caps)
    X_te_capped = apply_area_caps(X_te, area_caps)

    imputer = fit_numeric_imputer(X_tr_capped)

    X_tr_filled = apply_numeric_imputer(X_tr_capped, imputer)
    X_val_filled = apply_numeric_imputer(X_val_capped, imputer)
    X_te_filled = apply_numeric_imputer(X_te_capped, imputer)


    categories = fit_category_encoder(X_tr_filled)

    X_tr_encoded = apply_category_encoder(X_tr_filled, categories)
    X_val_encoded = apply_category_encoder(X_val_filled, categories)
    X_te_encoded = apply_category_encoder(X_te_filled, categories)

    X_tr_scaled, X_val_scaled, X_te_scaled = standardize(
        X_tr_encoded,
        X_val_encoded,
        X_te_encoded,
    )

    feature_names = X_tr_encoded.columns.tolist()

    varying = X_tr_encoded.to_numpy().std(axis=0) > 0

    print("\nScaled shapes:")
    print("Training:", X_tr_scaled.shape)
    print("Validation:", X_val_scaled.shape)
    print("Test:", X_te_scaled.shape)

    print(
        "\nTraining means approximately zero:",
        np.allclose(X_tr_scaled.mean(axis=0), 0, atol=1e-8),
    )

    print(
        "Nonconstant training standard deviations approximately one:",
        np.allclose(
            X_tr_scaled.std(axis=0)[varying],
            1,
            atol=1e-8,
        ),
    )

    for name, values in [
        ("Training", X_tr_scaled),
        ("Validation", X_val_scaled),
        ("Test", X_te_scaled),
    ]:
        print(f"{name} values finite:", np.isfinite(values).all())

    # print("\nCategories learned per feature:")
    # for column, values in categories.items():
    #     print(column, len(values))

    # print("\nEncoded shapes:")
    # print("Training:", X_tr_encoded.shape)
    # print("Validation:", X_val_encoded.shape)
    # print("Test:", X_te_encoded.shape)

    # same_columns = (
    #     X_tr_encoded.columns.equals(X_val_encoded.columns)
    #     and X_tr_encoded.columns.equals(X_te_encoded.columns)
    # )
    # print("\nIdentical feature columns:", same_columns)

    # for name, before, after in [
    #     ("Training", X_tr_filled, X_tr_encoded),
    #     ("Validation", X_val_filled, X_val_encoded),
    #     ("Test", X_te_filled, X_te_encoded),
    # ]:
    #     print(f"\n{name}:")
    #     print("All values finite:", np.isfinite(after.to_numpy()).all())
    #     print("Rows and order preserved:", before.index.equals(after.index))

    #     for column in CATEGORICAL_FEATURES:
    #         values = before[column].astype("string").fillna("unknown")
    #         unseen = ~values.isin(categories[column])
    #         print(f"Unseen {column} values:", unseen.sum())

    # print("\nOverall training medians:")
    # print(imputer["overall_medians"])

    # for name, before, after in [
    #     ("Training", X_tr_capped, X_tr_filled),
    #     ("Validation", X_val_capped, X_val_filled),
    #     ("Test", X_te_capped, X_te_filled),
    # ]:
    #     observed = before[NUMERIC_FEATURES].notna()

    #     changed_observed = (
    #         before[NUMERIC_FEATURES].ne(after[NUMERIC_FEATURES])
    #         & observed
    #     ).sum().sum()

    #     print(f"\n{name}:")
    #     print(
    #         "Numerical missing values remaining:",
    #         after[NUMERIC_FEATURES].isna().sum().sum(),
    #     )
    #     print("Observed values changed:", changed_observed)
    #     print("Rows and order preserved:", before.index.equals(after.index))
    #     print(
    #         "Missingness indicators preserved:",
    #         before[MISSING_INDICATORS].equals(after[MISSING_INDICATORS]),
    #     )




    # print("\nLearned area caps:")
    # for column, limits in area_caps.items():
    #     print(column, limits)

    # for name, original, capped in [
    #     ("Training", X_tr, X_tr_capped),
    #     ("Validation", X_val, X_val_capped),
    #     ("Test", X_te, X_te_capped),
    # ]:
    #     print(f"\n{name}:")
    #     print("Rows preserved:", len(original) == len(capped))

    #     for column in ["area_m2", "land_area_m2"]:
    #         changed = (
    #             original[column].notna()
    #             & original[column].ne(capped[column])
    #         )

    #         print(f"{column} values capped:", changed.sum())





    # for column in ["area_m2", "land_area_m2"]:
    #     summary = (
    #         X_tr.groupby("property_category")[column]
    #         .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    #     )

    #     print(f"\nTraining summary: {column}")
    #     print(
    #         summary[
    #             ["count", "min", "1%", "25%", "50%", "75%", "99%", "max"]
    #         ].round(2).to_string()
    #     )




    # print("\nSplit sizes:")
    # print("Training:", len(X_tr))
    # print("Validation:", len(X_val))
    # print("Test:", len(X_te))

    # print("\nTraining price threshold:", threshold)

    # print("\nPremium percentage:")
    # print("Training:", round(tier_tr.mean() * 100, 2))
    # print("Validation:", round(tier_val.mean() * 100, 2))
    # print("Test:", round(tier_te.mean() * 100, 2))

    # overlap = (
    #     set(X_tr.index) & set(X_val.index)
    #     | set(X_tr.index) & set(X_te.index)
    #     | set(X_val.index) & set(X_te.index)
    # )

    # print("\nOverlapping rows:", len(overlap))
    # print(
    #     "All rows accounted for:",
    #     len(X_tr) + len(X_val) + len(X_te) == len(X),
    # )

    # print("Raw rows:", len(raw_df))
    # print("Cleaned rows:", len(df))
    # print("Feature table shape:", X.shape)
    # print("Target shape:", y.shape)
    # print("Repeated URLs:", df["listing_url"].duplicated().sum())
    # print("Rows aligned:", X.index.equals(y.index))

    # rows_before = len(df)
    # rows_before_billing_filter = len(df)
    # print(
    #     "\nAmbiguous-price rows excluded:",
    #     rows_before_billing_filter - len(df),
    # )
    # print(
    #     "Flagged listings remaining:",
    #     df["listing_url"].isin(AMBIGUOUS_PRICE_LISTINGS).sum(),
    # )
    # print("Rows retained:", len(df))
    # print("Feature table shape:", X.shape)
    # print("Target shape:", y.shape)
    # remaining_leakage = [
    #     column for column in LEAKAGE_COLUMNS
    #     if column in df.columns
    # ]

    

    # review_urls = (
    #     raw_df.loc[
    #         raw_df["estate_id"].isin(BILLING_REVIEW_IDS),
    #         "estate_rel_url_x",
    #     ]
    #     .astype("string")
    #     .str.strip()
    #     .dropna()
    #     .unique()
    # )

    # print("\nListing URLs requiring billing review:")
    # print(review_urls)

    # review_rows = df.loc[
    #     df["listing_url"].isin(review_urls),
    #     [
    #         "listing_url",
    #         "property_category",
    #         "price",
    #         "description",
    #     ],
    # ]

    # for _, row in review_rows.iterrows():
    #     print("\nListing:", row["listing_url"])
    #     print("Category:", row["property_category"])
    #     print("Price:", row["price"])
    #     print("Description:", row["description"])

    # print("\nMissing coordinates after cleaning:")
    # print(df[["lat", "lng"]].isna().sum())

    # print("Coordinate indicators:", df["coordinates_missing"].sum())
    # print("Rows retained:", len(df))
    # print("Feature table shape:", X.shape)

    # print(
    #     df.loc[
    #         df["listing_url"].eq("/items/4596070"),
    #         ["listing_url", "city", "lat", "lng", "coordinates_missing"],
    #     ].to_string(index=False)
    # )




    # invalid_coordinates = (
    #     ~df["lat"].between(-90, 90)
    #     | ~df["lng"].between(-180, 180)
    # )

    # zero_coordinates = (
    #     df["lat"].eq(0)
    #     | df["lng"].eq(0)
    # )

    # print("\nMissing coordinates:")
    # print(df[["lat", "lng"]].isna().sum())

    # print("\nInvalid or missing coordinate rows:",
    #       invalid_coordinates.sum())

    # print("Rows with a zero coordinate:",
    #       zero_coordinates.sum())

    # print("\nCoordinate ranges:")
    # print(df[["lat", "lng"]].agg(["min", "max"]))

    # flagged = invalid_coordinates | zero_coordinates

    # print("\nFlagged coordinate examples:")
    # print(
    #     df.loc[
    #         flagged,
    #         ["listing_url", "city", "lat", "lng"],
    #     ].head(20).to_string(index=False)
    # )
    # coordinate_columns = [
    #     "listing_url",
    #     "city",
    #     "location",
    #     "lat",
    #     "lng",
    # ]

    # print("\n10 lowest latitudes:")
    # print(
    #     df.nsmallest(10, "lat")[coordinate_columns]
    #     .to_string(index=False)
    # )

    # print("\n10 lowest longitudes:")
    # print(
    #     df.nsmallest(10, "lng")[coordinate_columns]
    #     .to_string(index=False)
    # )

    # invalid_rooms = (
    #     df["rooms"].notna()
    #     & (
    #         (df["rooms"] <= 0)
    #         | (df["rooms"] % 1 != 0)
    #     )
    # )

    # print("\nInvalid room counts:", invalid_rooms.sum())

    # print(
    #     df.loc[
    #         invalid_rooms,
    #         ["listing_url", "property_category", "rooms", "area_m2"],
    #     ].head(20).to_string(index=False)
    # )

    # print("\nRoom-count frequencies:")
    # print(df["rooms"].value_counts(dropna=False).sort_index())

    # print("\nFeature table shape:", X.shape)
    # print("Target shape:", y.shape)
    # print("Selected features:", X.columns.tolist())

    # forbidden = ["price", "unit_price", "total_price"]
    # print(
    #     "Forbidden price columns in X:",
    #     [column for column in forbidden if column in X.columns],
    # )

    # print("Feature and target rows aligned:", X.index.equals(y.index))



    # print("\nLeakage columns still present:", remaining_leakage)
    # print("Target price retained:", "price" in df.columns)
    # print("Rows after dropping columns:", len(df))




    # print("\nRows before:", rows_before)
    # print("Rows after:", len(df))
    # print("Rows removed:", rows_before - len(df))
    # print("Repeated URLs remaining:", df["listing_url"].duplicated().sum())

    # print("\nRetained example listing:")
    # print(
    #     df.loc[
    #         df["listing_url"].eq("/items/4582627"),
    #         ["listing_url", "price", "datetime_scrape_x"],
    #     ].to_string(index=False)
    # )

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

    # print("\nMissingness indicators:")
    # print(
    #     df[
    #         [
    #             "land_area_m2_missing",
    #             "floor_number_missing",
    #             "total_floors_missing",
    #             "rooms_missing",
    #         ]
    #     ].sum()
    # )

    # print("\nRenovation categories:")
    # print(df["renovation"].value_counts(dropna=False))

    # print("\nMortgage categories:")
    # print(df["mortgage_available"].value_counts(dropna=False))


    # print("\nTotal rows:", len(df))
    # print("Exact duplicate rows beyond the first:", df.duplicated().sum())

    # print("\nMissing estate IDs:", df["estate_id"].isna().sum())

    # listing_counts = df["estate_id"].value_counts()
    # repeated_ids = listing_counts[listing_counts > 1]

    # print("Distinct non-missing estate IDs:", len(listing_counts))
    # print("Estate IDs appearing multiple times:", len(repeated_ids))
    # print("Extra rows for repeated IDs:", (repeated_ids - 1).sum())

    # print("\nFive most repeated estate IDs:")
    # print(repeated_ids.head(5))

    # example_ids = repeated_ids.head(2).index

    # print("\nExample repeated listings:")
    # print(
    #     df.loc[
    #         df["estate_id"].isin(example_ids),
    #         [
    #             "estate_id",
    #             "price",
    #             "property_category",
    #             "area_m2",
    #             "datetime_scrape_x",
    #             "datetime_scrape_y",
    #         ],
    #     ]
    #     .sort_values(["estate_id", "datetime_scrape_x"])
    #     .head(20)
    #     .to_string(index=False)
    # )





    # url_columns = [
    #     "estate_rel_url_x",
    #     "estate_rel_url_y",
    #     "estate_rel_url",
    # ]

    # for column in url_columns:
    #     urls = df[column].astype("string").str.strip()
    #     valid_urls = urls[urls.notna() & urls.ne("")]

    #     counts = valid_urls.value_counts()
    #     repeated = counts[counts > 1]

    #     print(f"\n--- {column} ---")
    #     print("Missing or blank URLs:", len(df) - len(valid_urls))
    #     print("Distinct URLs:", len(counts))
    #     print("URLs appearing multiple times:", len(repeated))
    #     print("Extra rows for repeated URLs:", (repeated - 1).sum())

    #     print("Example URLs:")
    #     print(valid_urls.head(3).to_string(index=False))

    #     print("Most repeated URLs:")
    #     print(repeated.head(5))

    # urls = df[
    #     ["estate_rel_url_x", "estate_rel_url_y", "estate_rel_url"]
    # ].apply(lambda column: column.astype("string").str.strip())

    # url_mismatch = (
    #     urls["estate_rel_url_x"].ne(urls["estate_rel_url_y"])
    #     | urls["estate_rel_url_x"].ne(urls["estate_rel_url"])
    # )

    # print("\nRows with conflicting listing URLs:", url_mismatch.sum())

    # scrape_time = pd.to_datetime(
    #     df["datetime_scrape_x"],
    #     format="mixed",
    #     utc=True,
    #     errors="coerce",
    # )

    # print("Missing or unreadable scrape timestamps:", scrape_time.isna().sum())

    # example_url = "/items/4582627"

    # example = df.loc[
    #     urls["estate_rel_url_x"].eq(example_url),
    #     ["estate_rel_url_x", "price", "area_m2", "datetime_scrape_x"],
    # ].copy()

    # example["parsed_time"] = scrape_time.loc[example.index]

    # print("\nHistory of one repeated listing:")
    # print(
    #     example.sort_values("parsed_time")
    #     .to_string(index=False)
    # )


