# Retail & Marketing Analytics Project
# Notebook 2: Data Cleaning and Preprocessing

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 02 - Data Cleaning and Preprocessing
Author: [Kritika Sehgal]

Objective:
- Handle missing values
- Remove duplicates
- Convert data types
- Detect and treat outliers
- Feature engineering
- Save cleaned dataset
"""
# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

RAW_DATA_PATH = Path("data/raw/retail_sales_data.csv")
PROCESSED_DATA_DIR = Path("data/processed")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
DOCS_DIR = Path("docs")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. LOAD DATASET
# =============================================================================

df_sales = pd.read_csv(RAW_DATA_PATH)

print("=" * 80)
print("DATA CLEANING & PREPROCESSING")
print("=" * 80)

print(f"\nInitial Shape      : {df_sales.shape}")
print(
    f"Memory Usage (MB) : "
    f"{df_sales.memory_usage(deep=True).sum() / 1024**2:.2f}"
)

# Create working copy
df_clean = df_sales.copy()

# =============================================================================
# 4. HANDLE MISSING VALUES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1 — HANDLE MISSING VALUES")
print("=" * 80)

missing_summary = (
    pd.DataFrame(
        {
            "Missing Values": df_clean.isna().sum(),
            "Missing (%)": (
                df_clean.isna().mean() * 100
            ).round(2),
        }
    )
    .query("`Missing Values` > 0")
    .sort_values("Missing Values", ascending=False)
)

if missing_summary.empty:
    print("\nNo missing values detected.")
else:
    print(missing_summary)

numeric_columns = df_clean.select_dtypes(
    include=np.number
).columns

categorical_columns = df_clean.select_dtypes(
    include="object"
).columns

print("\nApplying Missing Value Treatment...")

for column in numeric_columns:

    if df_clean[column].isna().sum() > 0:

        median_value = df_clean[column].median()

        df_clean[column] = df_clean[column].fillna(
            median_value
        )

        print(
            f"✓ {column:<25}"
            f"Median = {median_value:.2f}"
        )

for column in categorical_columns:

    if df_clean[column].isna().sum() > 0:

        mode = df_clean[column].mode()

        fill_value = (
            mode.iloc[0]
            if not mode.empty
            else "Unknown"
        )

        df_clean[column] = df_clean[column].fillna(
            fill_value
        )

        print(
            f"✓ {column:<25}"
            f"Mode = {fill_value}"
        )

print(
    f"\nRemaining Missing Values : "
    f"{df_clean.isna().sum().sum():,}"
)

# =============================================================================
# 5. REMOVE DUPLICATES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2 — REMOVE DUPLICATE RECORDS")
print("=" * 80)

duplicates_before = df_clean.duplicated().sum()

print(f"\nDuplicate Records Found : {duplicates_before:,}")

if duplicates_before > 0:

    df_clean = (
        df_clean
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"✓ Removed {duplicates_before:,} duplicate records."
    )

else:

    print("✓ No duplicate records detected.")

print(f"Updated Shape : {df_clean.shape}")

# =============================================================================
# 6. DATA TYPE OPTIMIZATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3 — DATA TYPE OPTIMIZATION")
print("=" * 80)

date_columns = [
    "Order_Date",
    "Ship_Date",
]

for column in date_columns:

    if column in df_clean.columns:

        df_clean[column] = pd.to_datetime(
            df_clean[column],
            errors="coerce",
        )

        print(f"✓ Converted {column} → datetime")

category_columns = [
    "Segment",
    "Region",
    "Product_Category",
    "Product_Sub_Category",
    "Order_Priority",
]

for column in category_columns:

    if column in df_clean.columns:

        df_clean[column] = (
            df_clean[column]
            .astype("category")
        )

        print(f"✓ Converted {column} → category")

if "Quantity" in df_clean.columns:

    df_clean["Quantity"] = (
        df_clean["Quantity"]
        .astype("int32")
    )

    print("✓ Optimized Quantity → int32")

memory_before = (
    df_sales.memory_usage(deep=True).sum()
    / 1024**2
)

memory_after = (
    df_clean.memory_usage(deep=True).sum()
    / 1024**2
)

print("\nMemory Optimization Summary")
print("-" * 40)
print(f"Before : {memory_before:.2f} MB")
print(f"After  : {memory_after:.2f} MB")
print(
    f"Saved  : "
    f"{memory_before - memory_after:.2f} MB"
)

print("\n✓ Missing value treatment completed.")
print("✓ Duplicate removal completed.")
print("✓ Data types optimized.")

# =============================================================================
# 7. OUTLIER DETECTION & TREATMENT
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4 — OUTLIER DETECTION & TREATMENT")
print("=" * 80)


def detect_outliers_iqr(dataframe, column):
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Returns
    -------
    tuple
        Outlier dataframe, lower bound, upper bound.
    """

    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = dataframe[
        (dataframe[column] < lower_bound)
        | (dataframe[column] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


def save_figure(filename):
    """Save current matplotlib figure."""

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


outlier_columns = [
    column
    for column in ["Sales", "Quantity", "Profit"]
    if column in df_clean.columns
]

print("\nOutlier Summary")

outlier_summary = []

for column in outlier_columns:

    outliers, lower_bound, upper_bound = detect_outliers_iqr(
        df_clean,
        column,
    )

    percentage = (len(outliers) / len(df_clean)) * 100

    outlier_summary.append(
        {
            "Column": column,
            "Outliers": len(outliers),
            "Percentage (%)": round(percentage, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
        }
    )

outlier_summary = pd.DataFrame(outlier_summary)

print(outlier_summary)

# =============================================================================
# VISUALIZATION — BEFORE TREATMENT
# =============================================================================

fig, axes = plt.subplots(
    1,
    len(outlier_columns),
    figsize=(6 * len(outlier_columns), 5),
)

if len(outlier_columns) == 1:
    axes = [axes]

for axis, column in zip(axes, outlier_columns):

    axis.boxplot(df_clean[column].dropna())

    axis.set_title(
        f"{column}\nBefore Treatment",
        fontsize=12,
        fontweight="bold",
    )

    axis.set_ylabel(column)

    axis.grid(alpha=0.3)

save_figure("03_outliers_before_treatment.png")

# =============================================================================
# OUTLIER TREATMENT (WINSORIZATION)
# =============================================================================

print("\nApplying Winsorization...")

for column in outlier_columns:

    outliers, lower_bound, upper_bound = detect_outliers_iqr(
        df_clean,
        column,
    )

    if len(outliers) == 0:

        print(f"✓ {column:<15} No outliers detected")

        continue

    df_clean[f"{column}_Original"] = df_clean[column]

    df_clean[column] = df_clean[column].clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    print(
        f"✓ {column:<15}"
        f"Capped to [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

# =============================================================================
# VISUALIZATION — AFTER TREATMENT
# =============================================================================

fig, axes = plt.subplots(
    1,
    len(outlier_columns),
    figsize=(6 * len(outlier_columns), 5),
)

if len(outlier_columns) == 1:
    axes = [axes]

for axis, column in zip(axes, outlier_columns):

    axis.boxplot(df_clean[column].dropna())

    axis.set_title(
        f"{column}\nAfter Treatment",
        fontsize=12,
        fontweight="bold",
    )

    axis.set_ylabel(column)

    axis.grid(alpha=0.3)

save_figure("04_outliers_after_treatment.png")

print("\n✓ Outlier treatment completed successfully.")

# =============================================================================
# 8. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5 — FEATURE ENGINEERING")
print("=" * 80)

# -----------------------------------------------------------------------------
# Time-Based Features
# -----------------------------------------------------------------------------

if "Order_Date" in df_clean.columns:

    print("\nCreating Time-Based Features...")

    df_clean["Year"] = df_clean["Order_Date"].dt.year
    df_clean["Quarter"] = df_clean["Order_Date"].dt.quarter
    df_clean["Month"] = df_clean["Order_Date"].dt.month
    df_clean["Month_Name"] = df_clean["Order_Date"].dt.month_name()

    df_clean["Day"] = df_clean["Order_Date"].dt.day
    df_clean["Day_of_Week"] = df_clean["Order_Date"].dt.dayofweek
    df_clean["Day_Name"] = df_clean["Order_Date"].dt.day_name()

    df_clean["Week_of_Year"] = (
        df_clean["Order_Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df_clean["Is_Weekend"] = (
        df_clean["Day_of_Week"]
        .isin([5, 6])
        .astype(int)
    )

    df_clean["Is_Month_Start"] = (
        df_clean["Order_Date"]
        .dt.is_month_start
        .astype(int)
    )

    df_clean["Is_Month_End"] = (
        df_clean["Order_Date"]
        .dt.is_month_end
        .astype(int)
    )

    print("✓ Time features created.")

# -----------------------------------------------------------------------------
# Revenue Features
# -----------------------------------------------------------------------------

if {"Sales", "Quantity"}.issubset(df_clean.columns):

    print("\nCreating Revenue Features...")

    df_clean["Unit_Price"] = (
        df_clean["Sales"] /
        df_clean["Quantity"]
    )

    df_clean["Revenue"] = df_clean["Sales"]

    if "Discount" in df_clean.columns:

        df_clean["Discount_Amount"] = (
            df_clean["Sales"] *
            df_clean["Discount"]
        )

        df_clean["Net_Revenue"] = (
            df_clean["Revenue"] -
            df_clean["Discount_Amount"]
        )

    if "Profit" in df_clean.columns:

        df_clean["Profit_Margin"] = (
            df_clean["Profit"] /
            df_clean["Sales"]
        ) * 100

        df_clean["Profit_Ratio"] = (
            df_clean["Profit"] /
            df_clean["Sales"]
        )

    print("✓ Revenue features created.")

# -----------------------------------------------------------------------------
# Delivery Features
# -----------------------------------------------------------------------------

if {"Order_Date", "Ship_Date"}.issubset(df_clean.columns):

    print("\nCreating Delivery Features...")

    df_clean["Delivery_Days"] = (
        df_clean["Ship_Date"] -
        df_clean["Order_Date"]
    ).dt.days

    df_clean["Delivery_Category"] = pd.cut(
        df_clean["Delivery_Days"],
        bins=[-np.inf, 2, 5, 10, np.inf],
        labels=[
            "Same/Next Day",
            "Standard",
            "Delayed",
            "Very Delayed",
        ],
    )

    print("✓ Delivery features created.")

# -----------------------------------------------------------------------------
# Customer Features
# -----------------------------------------------------------------------------

if "Customer_ID" in df_clean.columns:

    print("\nCreating Customer Features...")

    customer_orders = (
        df_clean
        .groupby("Customer_ID")
        .size()
    )

    df_clean["Customer_Order_Count"] = (
        df_clean["Customer_ID"]
        .map(customer_orders)
    )

    df_clean["Is_Repeat_Customer"] = (
        df_clean["Customer_Order_Count"] > 1
    ).astype(int)

    print("✓ Customer features created.")

# -----------------------------------------------------------------------------
# Product Features
# -----------------------------------------------------------------------------

if "Product_ID" in df_clean.columns:

    print("\nCreating Product Features...")

    product_summary = (
        df_clean
        .groupby("Product_ID")["Sales"]
        .agg(
            Product_Total_Sales="sum",
            Product_Average_Sales="mean",
            Product_Order_Count="count",
        )
    )

    df_clean = df_clean.merge(
        product_summary,
        left_on="Product_ID",
        right_index=True,
        how="left",
    )

    print("✓ Product features created.")

# -----------------------------------------------------------------------------
# Sales Category
# -----------------------------------------------------------------------------

if "Sales" in df_clean.columns:

    df_clean["Sales_Category"] = pd.qcut(
        df_clean["Sales"],
        q=4,
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
        duplicates="drop",
    )

    print("✓ Sales category created.")

# -----------------------------------------------------------------------------
# Season Feature
# -----------------------------------------------------------------------------

if "Month" in df_clean.columns:

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }

    df_clean["Season"] = (
        df_clean["Month"]
        .map(season_map)
    )

    print("✓ Season feature created.")

print("\nFeature Engineering Completed.")
print(f"Current Dataset Shape : {df_clean.shape}")

# =============================================================================
# 9. DATA VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6 — DATA VALIDATION")
print("=" * 80)

validation_results = []

if "Sales" in df_clean.columns:

    validation_results.append(
        {
            "Validation": "Negative Sales",
            "Count": (df_clean["Sales"] < 0).sum(),
        }
    )

if "Quantity" in df_clean.columns:

    validation_results.append(
        {
            "Validation": "Zero/Negative Quantity",
            "Count": (df_clean["Quantity"] <= 0).sum(),
        }
    )

if "Discount" in df_clean.columns:

    validation_results.append(
        {
            "Validation": "Invalid Discount",
            "Count": (
                (df_clean["Discount"] < 0)
                | (df_clean["Discount"] > 1)
            ).sum(),
        }
    )

if {"Order_Date", "Ship_Date"}.issubset(df_clean.columns):

    validation_results.append(
        {
            "Validation": "Ship Date Before Order Date",
            "Count": (
                df_clean["Ship_Date"]
                <
                df_clean["Order_Date"]
            ).sum(),
        }
    )

validation_summary = pd.DataFrame(validation_results)

print(validation_summary)

print("\n✓ Data validation completed successfully.")

# =============================================================================
# 10. CLEANED DATA SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("CLEANED DATA SUMMARY")
print("=" * 80)

records_removed = df_sales.shape[0] - df_clean.shape[0]
features_added = df_clean.shape[1] - df_sales.shape[1]

memory_before = df_sales.memory_usage(deep=True).sum() / 1024**2
memory_after = df_clean.memory_usage(deep=True).sum() / 1024**2

print(f"\nFinal Dataset Shape : {df_clean.shape}")
print(f"Records Removed     : {records_removed:,}")
print(f"Features Added      : {features_added}")
print(f"Memory Usage        : {memory_after:.2f} MB")

print("\nData Quality Metrics")
print("-" * 40)

print(f"Missing Values      : {df_clean.isna().sum().sum():,}")
print(f"Duplicate Records   : {df_clean.duplicated().sum():,}")

complete_rows = df_clean.dropna().shape[0]

print(
    f"Complete Records    : "
    f"{complete_rows:,} "
    f"({complete_rows / len(df_clean) * 100:.2f}%)"
)

print("\nBusiness Summary")
print("-" * 40)

if "Sales" in df_clean.columns:
    print(f"Total Sales         : ${df_clean['Sales'].sum():,.2f}")
    print(f"Average Order Value : ${df_clean['Sales'].mean():,.2f}")

if "Customer_ID" in df_clean.columns:
    print(
        f"Unique Customers    : "
        f"{df_clean['Customer_ID'].nunique():,}"
    )

if "Product_ID" in df_clean.columns:
    print(
        f"Unique Products     : "
        f"{df_clean['Product_ID'].nunique():,}"
    )

# =============================================================================
# 11. EXPORT CLEANED DATA
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7 — EXPORT DATASETS")
print("=" * 80)

clean_data_path = PROCESSED_DATA_DIR / "cleaned_retail_sales.csv"

df_clean.to_csv(
    clean_data_path,
    index=False,
)

print(f"✓ Cleaned dataset saved to:\n{clean_data_path}")

# =============================================================================
# 12. GENERATE DATA DICTIONARY
# =============================================================================

data_dictionary = pd.DataFrame(
    {
        "Column": df_clean.columns,
        "Data Type": df_clean.dtypes.astype(str),
        "Non-Null Count": df_clean.count().values,
        "Null Count": df_clean.isna().sum().values,
        "Unique Values": [
            df_clean[column].nunique()
            for column in df_clean.columns
        ],
        "Sample Value": [
            (
                df_clean[column].iloc[0]
                if not df_clean.empty
                else None
            )
            for column in df_clean.columns
        ],
    }
)

dictionary_path = DOCS_DIR / "data_dictionary.csv"

data_dictionary.to_csv(
    dictionary_path,
    index=False,
)

print(f"✓ Data dictionary saved to:\n{dictionary_path}")

# =============================================================================
# 13. GENERATE CLEANING REPORT
# =============================================================================

quality_score = (
    1
    - (
        df_clean.isna().sum().sum()
        /
        (df_clean.shape[0] * df_clean.shape[1])
    )
) * 100

report = f"""
================================================================================
RETAIL & MARKETING ANALYTICS PROJECT
DATA CLEANING REPORT
================================================================================

DATA QUALITY SUMMARY
--------------------
Original Records      : {df_sales.shape[0]:,}
Final Records         : {df_clean.shape[0]:,}
Records Removed       : {records_removed:,}

Missing Values Before : {df_sales.isna().sum().sum():,}
Missing Values After  : {df_clean.isna().sum().sum():,}

Duplicate Rows Before : {duplicates_before:,}
Duplicate Rows After  : {df_clean.duplicated().sum():,}

Data Quality Score    : {quality_score:.2f}%

FEATURE ENGINEERING
-------------------
Original Features     : {df_sales.shape[1]}
New Features Added    : {features_added}
Final Features        : {df_clean.shape[1]}

New Feature Categories
• Time-based Features
• Revenue Metrics
• Customer Metrics
• Product Metrics
• Delivery Metrics
• Seasonal Features

OUTLIER TREATMENT
-----------------
Method Used           : IQR Winsorization
Columns Processed     : {", ".join(outlier_columns)}

MEMORY OPTIMIZATION
-------------------
Memory Before         : {memory_before:.2f} MB
Memory After          : {memory_after:.2f} MB
Memory Saved          : {memory_before - memory_after:.2f} MB

OUTPUT FILES
------------
✓ cleaned_retail_sales.csv
✓ data_dictionary.csv
✓ cleaning_report.txt

NEXT STEP
---------
Proceed to:
Notebook 03 – Exploratory Data Analysis

Generated On:
{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

================================================================================
"""

report_path = REPORTS_DIR / "02_cleaning_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

print(f"✓ Cleaning report saved to:\n{report_path}")

# =============================================================================
# 14. NOTEBOOK COMPLETION
# =============================================================================

print("\n" + "=" * 80)
print("NOTEBOOK 02 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    f"""
Summary
-------
✓ Missing values handled
✓ Duplicate records removed
✓ Data types optimized
✓ Outliers detected and treated
✓ Feature engineering completed
✓ Data validation completed
✓ Clean dataset exported
✓ Data dictionary generated
✓ Cleaning report generated

Output Files
------------
• {clean_data_path}
• {dictionary_path}
• {report_path}

Next Notebook
-------------
Notebook 03 — Exploratory Data Analysis (EDA)
"""
)