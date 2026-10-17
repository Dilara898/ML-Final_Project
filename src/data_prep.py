"""
Load and clean property listings, split the data, and prepare model features.

Learn preprocessing statistics and the price-tier threshold from training
data only.
"""
# Price audit:
# All 100,775 rows have positive, non-missing prices and AZN currency.
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


def make_features(df, preprocessing=None):
    """
    Build numerical features and return (X, y, feature_names).

    df must already be cleaned.
    preprocessing must contain settings fitted on training data.
    """
    if preprocessing is None:
        raise ValueError(
            "Pass training-fitted preprocessing. "
            "Split the data before building numerical features."
        )

    X, y = select_features(df)

    X = apply_area_caps(X, preprocessing["area_caps"])
    X = apply_numeric_imputer(X, preprocessing["imputer"])
    X = apply_category_encoder(X, preprocessing["categories"])

    return X, y, X.columns.tolist()

def prepare_data(path=DATA_PATH, seed=SEED):
    """Run cleaning, splitting, feature preparation, and scaling."""
    raw_df = load_raw(path)
    df = clean(raw_df)

    X, y = select_features(df)

    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(
        X, y, seed=seed
    )

    tier_tr, threshold = make_tier_label(y_tr)
    tier_val, _ = make_tier_label(y_val, threshold=threshold)
    tier_te, _ = make_tier_label(y_te, threshold=threshold)

    # Learn each preprocessing step from training data only.
    area_caps = fit_area_caps(X_tr)
    X_tr_capped = apply_area_caps(X_tr, area_caps)

    imputer = fit_numeric_imputer(X_tr_capped)
    X_tr_filled = apply_numeric_imputer(X_tr_capped, imputer)

    categories = fit_category_encoder(X_tr_filled)

    preprocessing = {
        "area_caps": area_caps,
        "imputer": imputer,
        "categories": categories,
    }

    # Apply the same learned settings to every split.
    X_tr_encoded, _, feature_names = make_features(
        df.loc[X_tr.index], preprocessing
    )
    X_val_encoded, _, _ = make_features(
        df.loc[X_val.index], preprocessing
    )
    X_te_encoded, _, _ = make_features(
        df.loc[X_te.index], preprocessing
    )

    X_tr_scaled, X_val_scaled, X_te_scaled = standardize(
        X_tr_encoded, X_val_encoded, X_te_encoded
    )

    return {
        "X_train": X_tr_encoded,
        "X_val": X_val_encoded,
        "X_test": X_te_encoded,
        "X_train_scaled": X_tr_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_te_scaled,
        "y_train": y_tr,
        "y_val": y_val,
        "y_test": y_te,
        "tier_train": tier_tr,
        "tier_val": tier_val,
        "tier_test": tier_te,
        "tier_threshold": threshold,
        "feature_names": feature_names,
        "preprocessing": preprocessing,
        "training_df": df.loc[X_tr.index].copy(),
    }


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
    data = prepare_data()

    for split in ["train", "val", "test"]:
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        tiers = data[f"tier_{split}"]
        scaled = data[f"X_{split}_scaled"]

        print(f"\n{split.upper()}")
        print("Feature shape:", X.shape)
        print("Rows aligned:", X.index.equals(y.index))
        print("Tier count matches:", len(tiers) == len(y))
        print("Encoded values finite:", np.isfinite(X.to_numpy()).all())
        print("Scaled values finite:", np.isfinite(scaled).all())

    print("\nTier threshold:", data["tier_threshold"])
    print("Feature count:", len(data["feature_names"]))
