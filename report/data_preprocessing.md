# Data preprocessing

## Dataset

We used the bina.az sale dataset, published on Kaggle as
"binaaz-sale-project" by sehriyarmemmedli.

Local filename: data/house_sale.csv

Dataset version: TO BE RECORDED
Download date: TO BE RECORDED

The raw dataset contained 100,775 rows and 51 columns.
The CSV is excluded from version control.

## Column translation and numerical parsing

Azerbaijani column names were translated into English through an
explicit mapping in src/data_prep.py.

Floor strings such as "7 / 9" were separated into floor_number
and total_floors.

Area values were converted into square metres. Values recorded
in sot were multiplied by 100.

Missing source values were preserved during parsing. The audited
dataset had no parsing failures among non-missing floor, area,
or land-area values.

## Currency and sale-price interpretation

Both currency columns contained AZN throughout the audited dataset.
All recorded prices were positive and non-missing.

The price column was retained as the target. It differed from
total_price in 13 raw rows. These rows had different collection
timestamps; we did not overwrite the target.

Rental keywords were not used as an automatic exclusion rule.
Sale descriptions may mention rental income, existing tenants,
or parking fees.

Two manually reviewed listings were excluded because the meaning
of their sale-price target was ambiguous:

- /items/4656308
- /items/4604180

No daily-to-monthly conversion was applied to sale prices.
The description review was not exhaustive.

## Duplicate listings

Unique estate_id values did not identify unique property listings.
Repeated observations were identified using the listing URL.

For each URL, we retained the row with the latest
datetime_scrape_x timestamp. Timestamps without timezone
information were interpreted as UTC.

For equal timestamps, the last row in original input order
was retained.

This removed 36,321 repeated observations, leaving 64,454 rows.
After excluding the two ambiguous-price listings, 64,452 rows remained.

Different URLs may still refer to the same physical property.

## Coordinates

The coordinates for /items/4596070 conflicted with its stated
Baku location. Both latitude and longitude were marked as missing,
while the listing itself was retained.

A coordinates_missing indicator records missing or unreliable
coordinate pairs.

Global coordinate-range checks do not establish that every
remaining coordinate matches its stated address.

## Missing values

Missing renovation and mortgage information was represented
as "unknown", rather than assumed to mean "no".

Missingness indicators were created for land area, floor number,
total floors, rooms, and coordinates.

Numerical missing values were filled using training medians
within property category. When a category had no observed values,
the overall training median was used.

Many missing values reflect property type. For example, apartment
floor information is not recorded for land listings. Filled values
are modelling placeholders, not verified property measurements.

The building_type column was excluded because approximately
99.85% of its raw values were missing.

## Leakage prevention and feature selection

The following price-related columns were removed:

- unit_price
- total_price

Model inputs were selected through an explicit feature list.
Descriptions, URLs, IDs, owner details, and other unselected
columns were not included as model features.

The selected table contained 18 columns before categorical encoding.

## Train, validation, and test split

The random seed was 42.

Training rows were selected randomly first, using approximately
70% of the cleaned dataset. The price-tier threshold was then
calculated from the training median.

The remaining rows were divided into validation and test sets
using stratification by the resulting price-tier labels.

Split sizes:

- Training: 45,116
- Validation: 9,667
- Test: 9,669

The training median was 210,000 AZN.

- Standard: price <= 210,000 AZN
- Premium: price > 210,000 AZN

No listing URL appeared in more than one split.

## Outlier handling

Upper area limits were learned from training data separately
for each property category.

For area_m2 and land_area_m2, values above the category's
99th percentile were capped at that percentile.

A cap was learned only when a category contained at least
100 observed training values for that feature.

Categories with fewer observations were left uncapped.
The same learned limits were applied to validation and test.

No rows were removed by this rule, and target prices were
not capped.

Capping reduces the influence of extreme areas but can also
reduce differences between genuinely large properties.
It does not establish whether a source value is incorrect.

## Categorical encoding and standardization

One-hot encoding categories were learned from training data only.
Previously unseen categories produced an all-zero encoding
within the relevant feature's category columns.

The resulting model table contained 173 numerical features.

Standardization used training means and standard deviations
for all encoded features. A divisor of 1 was used for columns
with zero training standard deviation.

Validation and test data used the same training statistics.

## Cross-validation handoff

The processed X_train table must not be treated as raw input
for cross-validation.

For each cross-validation round, caps, imputation values,
encoding categories, and scaling statistics must be learned
from that round's training folds only.

The cleaned training_df returned by prepare_data() is available
as the starting point before these learned transformations.

## EDA

Price, area, rooms, and geographic figures use cleaned training
data before area capping and numerical imputation.

Missing room counts and coordinates are excluded only from
the relevant plots, rather than filled with modelling values.

The class-balance figure shows label counts across all three
splits using the same training-derived threshold.





## EDA findings and limitations

### Price distribution

Sale prices have a long right tail: a small number of expensive
listings extend far beyond the bulk of the data.

The natural-log price plot makes the distribution of typical
listings easier to inspect. This visualization alone does not
establish that prices follow a log-normal distribution.

### Area versus price

The plots suggest a positive association between area and price,
with substantial variation within property categories.

Property categories are shown separately because apartment floor
area and land-plot area represent different kinds of properties.

Both axes use logarithmic scales. These plots use areas before
capping, so extreme values remain visible.

Some unusually low prices and large areas remain in the dataset.
Their presence does not prove an error, but they limit confidence
in individual observations.

### Rooms

Two-room and three-room properties account for most observed
residential training listings.

Median prices generally increase with room count, although the
relationship varies by property category and is not perfectly
monotonic.

Some groups contain few observations. For older-build apartments,
the six-room group contains 25 listings, and the seven-or-more-room
group contains nine. Their medians should be interpreted cautiously.

These comparisons do not isolate the effect of room count:
area, location, renovation, and other characteristics also differ
between properties.

### Geographic distribution

The training data is heavily concentrated in listings labelled
Bakı. Results therefore primarily describe this market and may
not generalize well to other cities.

The geographic plots show observed coordinates, with price
represented using a shared logarithmic colour scale.

One training listing with unreliable coordinates is omitted from
these plots. Imputed coordinates are not plotted.

The figures have no background map and do not verify administrative
boundaries or address accuracy.

### Price-tier balance

The training-median threshold of 210,000 AZN produces nearly
balanced classes:

| Split | Standard | Premium | Premium percentage |
|---|---:|---:|---:|
| Training | 22,840 | 22,276 | 49.37% |
| Validation | 4,887 | 4,780 | 49.45% |
| Test | 4,888 | 4,781 | 49.45% |

The classes are not exactly equal because prices equal to the
threshold belong to the Standard class.

These labels represent relative price tiers in this dataset.
They do not independently establish property quality or luxury.

### General limitations

Listing prices are advertised asking prices, not verified
transaction prices.

The description review and coordinate checks were not exhaustive.
Ambiguous prices, incorrect measurements, and repeated physical
properties under different URLs may remain.

Random splitting evaluates performance on held-out listings from
the collected dataset. It does not directly measure performance
on future market conditions.