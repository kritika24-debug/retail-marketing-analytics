# Retail & Marketing Analytics Project
# Notebook 1: Data Acquisition and Setup

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 01 - Data Acquisition and Setup
Author: [Kritika Sehgal]

Objective:
- Setup project environment
- Download dataset from Kaggle
- Initial data inspection
- Create project folder structure
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

from pathlib import Path
import shutil
import warnings

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

warnings.filterwarnings("ignore")

# Visualization settings
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Pandas display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.2f}".format)

# Project paths
BASE_DIR = Path(".")
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

DATASET_NAME = (
    "abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing"
)

DATA_FILE = RAW_DATA_DIR / "retail_sales_data.csv"

print("=" * 80)
print("Retail & Marketing Analytics Project")
print("=" * 80)
print(f"Pandas Version : {pd.__version__}")
print(f"NumPy Version  : {np.__version__}")

# =============================================================================
# 3. CREATE PROJECT STRUCTURE
# =============================================================================

PROJECT_FOLDERS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "scripts",
    "dashboards",
    "outputs/figures",
    "outputs/reports",
    "docs",
]


def create_project_structure(folder_list):
    """
    Create project directory structure.
    """
    for folder in folder_list:
        Path(folder).mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Created {len(folder_list)} project directories.")


create_project_structure(PROJECT_FOLDERS)

# =============================================================================
# 4. DOWNLOAD DATASET FROM KAGGLE
# =============================================================================

def download_dataset():
    """
    Download dataset from KaggleHub and save it to the raw data folder.
    """
    print("\nDownloading dataset from Kaggle...")

    download_path = kagglehub.dataset_download(DATASET_NAME)

    csv_files = list(Path(download_path).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in downloaded dataset."
        )

    destination = RAW_DATA_DIR / "retail_sales_data.csv"

    shutil.copy(csv_files[0], destination)

    print(f"✓ Dataset saved to: {destination}")

    return destination


if not DATA_FILE.exists():
    download_dataset()
else:
    print(f"\n✓ Dataset already exists: {DATA_FILE}")

# =============================================================================
# 5. LOAD DATASET
# =============================================================================

def load_dataset(filepath):
    """
    Load dataset from CSV file.

    Parameters
    ----------
    filepath : Path or str

    Returns
    -------
    pandas.DataFrame
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}"
        )

    df = pd.read_csv(filepath)

    print("\n✓ Dataset loaded successfully")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


df_sales = load_dataset(DATA_FILE)

# =============================================================================
# 6. INITIAL DATA INSPECTION
# =============================================================================

print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print(f"\nDataset Shape: {df_sales.shape}")

print("\nFirst 5 Rows:")
print(df_sales.head())

print("\nLast 5 Rows:")
print(df_sales.tail())

print("\nDataset Information:")
df_sales.info()

print("\nData Types:")
print(df_sales.dtypes.to_frame(name="Data Type"))

print("\nMemory Usage:")
memory_usage = df_sales.memory_usage(deep=True).sum() / 1024**2
print(f"{memory_usage:.2f} MB")

# =============================================================================
# 7. DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

print("\nNumerical Summary")
print(df_sales.describe().T)

print("\nCategorical Summary")
print(df_sales.describe(include=["object"]).T)

print("\nUnique Values by Column")

unique_summary = (
    pd.DataFrame(
        {
            "Unique Values": df_sales.nunique(),
            "Missing Values": df_sales.isna().sum(),
            "Data Type": df_sales.dtypes.astype(str),
        }
    )
    .sort_values("Unique Values", ascending=False)
)

print(unique_summary)

# =============================================================================
# 8. DATA QUALITY ASSESSMENT
# =============================================================================

print("\n" + "=" * 80)
print("DATA QUALITY ASSESSMENT")
print("=" * 80)

missing_summary = (
    pd.DataFrame(
        {
            "Missing Values": df_sales.isna().sum(),
            "Missing (%)": (
                df_sales.isna().mean() * 100
            ).round(2),
        }
    )
    .query("`Missing Values` > 0")
    .sort_values("Missing Values", ascending=False)
)

duplicates = df_sales.duplicated().sum()

print(f"\nDuplicate Records : {duplicates:,}")
print(
    f"Duplicate Percentage : "
    f"{duplicates / len(df_sales) * 100:.2f}%"
)

if missing_summary.empty:
    print("\nNo missing values found.")
else:
    print("\nMissing Values Summary")
    print(missing_summary)

# =============================================================================
# 9. CREATE OUTPUT DIRECTORIES
# =============================================================================

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 10. HELPER FUNCTION
# =============================================================================

def save_figure(filename):
    """
    Save the current matplotlib figure.
    """
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# =============================================================================
# 11. VISUALIZATION — MISSING VALUES
# =============================================================================

missing_plot = (
    df_sales.isna()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

missing_plot.plot(
    kind="bar",
    color="coral",
)

plt.title(
    "Top 10 Columns with Missing Values",
    fontsize=14,
    weight="bold",
)

plt.xlabel("Columns")
plt.ylabel("Missing Values")

save_figure("01_missing_values.png")

# =============================================================================
# 12. VISUALIZATION — DATA TYPES
# =============================================================================

dtype_counts = df_sales.dtypes.value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    dtype_counts.values,
    labels=dtype_counts.index.astype(str),
    autopct="%1.1f%%",
    startangle=90,
)

plt.title(
    "Distribution of Data Types",
    fontsize=14,
    weight="bold",
)

save_figure("02_data_types_distribution.png")

# =============================================================================
# 13. GENERATE DATA INSPECTION REPORT
# =============================================================================

report = f"""
================================================================================
RETAIL & MARKETING ANALYTICS PROJECT
INITIAL DATA INSPECTION REPORT
================================================================================

DATASET OVERVIEW
----------------
Rows                : {df_sales.shape[0]:,}
Columns             : {df_sales.shape[1]}
Memory Usage (MB)   : {memory_usage:.2f}

DATA QUALITY
------------
Missing Cells       : {df_sales.isna().sum().sum():,}
Duplicate Records   : {duplicates:,}
Complete Records    : {df_sales.dropna().shape[0]:,}

COLUMN SUMMARY
--------------
Numerical Columns   : {len(df_sales.select_dtypes(include=np.number).columns)}
Categorical Columns : {len(df_sales.select_dtypes(include="object").columns)}
Datetime Columns    : {len(df_sales.select_dtypes(include="datetime").columns)}

NEXT STEPS
----------
1. Data Cleaning
2. Missing Value Treatment
3. Duplicate Removal
4. Data Type Conversion
5. Outlier Detection
6. Feature Engineering

Report Generated:
{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

================================================================================
"""

report_path = REPORTS_DIR / "01_initial_inspection_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

print("\n✓ Inspection report saved.")
print(f"Location: {report_path}")

# =============================================================================
# 14. SAVE CHECKPOINT
# =============================================================================

checkpoint_path = RAW_DATA_DIR / "original_data_checkpoint.csv"

df_sales.to_csv(
    checkpoint_path,
    index=False,
)

print(f"\n✓ Dataset checkpoint saved to:\n{checkpoint_path}")

# =============================================================================
# NOTEBOOK COMPLETED
# =============================================================================

print("\n" + "=" * 80)
print("NOTEBOOK 01 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"""
Summary
-------
✓ Project structure created
✓ Dataset downloaded
✓ Dataset loaded
✓ Initial inspection completed
✓ Data quality assessment completed
✓ Visualizations generated
✓ Inspection report exported
✓ Dataset checkpoint saved

Next Notebook:
Notebook 02 — Data Cleaning & Preprocessing
""")