# Retail & Marketing Analytics Project
# Notebook 3: Exploratory Data Analysis (EDA)

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 03 - Exploratory Data Analysis (EDA)
Author: [Kritika Sehgal]

Objective:
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Time series analysis
- Customer behavior analysis
- Product performance analysis
- Generate insights
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

DATA_PATH = Path("data/processed/cleaned_retail_sales.csv")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. LOAD DATASET
# =============================================================================

df = pd.read_csv(DATA_PATH)

for column in ["Order_Date", "Ship_Date"]:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column])

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\nDataset Shape : {df.shape}")

if "Order_Date" in df.columns:
    print(
        f"Analysis Period : "
        f"{df['Order_Date'].min().date()} "
        f"to "
        f"{df['Order_Date'].max().date()}"
    )

# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def save_figure(filename):
    """Save current matplotlib figure."""

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def print_separator(title):
    """Display formatted section heading."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# 5. DATASET OVERVIEW
# =============================================================================

print_separator("DATASET OVERVIEW")

print("\nFirst Five Records")
print(df.head())

print("\nDataset Information")
df.info()

print("\nMissing Values")
print(df.isna().sum().to_frame("Missing Values"))

print("\nDescriptive Statistics")
print(df.describe().T)

# =============================================================================
# 6. UNIVARIATE ANALYSIS — NUMERICAL VARIABLES
# =============================================================================

print_separator("UNIVARIATE ANALYSIS — NUMERICAL VARIABLES")

numerical_columns = [
    column
    for column in [
        "Sales",
        "Quantity",
        "Profit",
        "Discount",
        "Unit_Price",
    ]
    if column in df.columns
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
)

axes = axes.flatten()

for axis, column in zip(axes, numerical_columns):

    axis.hist(
        df[column],
        bins=40,
        edgecolor="black",
        alpha=0.75,
    )

    axis.axvline(
        df[column].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean : {df[column].mean():.2f}",
    )

    axis.axvline(
        df[column].median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median : {df[column].median():.2f}",
    )

    axis.set_title(
        column,
        fontsize=12,
        fontweight="bold",
    )

    axis.set_xlabel(column)
    axis.set_ylabel("Frequency")

    axis.grid(alpha=0.3)

    axis.legend(fontsize=8)

for axis in axes[len(numerical_columns):]:
    fig.delaxes(axis)

save_figure("05_numerical_distributions.png")

print("\nNumerical Summary")
print(df[numerical_columns].describe().round(2).T)

print("\nDistribution Metrics")

distribution_summary = pd.DataFrame(
    {
        "Mean": df[numerical_columns].mean(),
        "Median": df[numerical_columns].median(),
        "Std Dev": df[numerical_columns].std(),
        "Skewness": df[numerical_columns].skew(),
        "Kurtosis": df[numerical_columns].kurt(),
    }
).round(2)

print(distribution_summary)

print("\nBusiness Insights")
print("-" * 40)

for column in numerical_columns:

    skewness = df[column].skew()

    if skewness > 1:
        interpretation = "Highly Right Skewed"

    elif skewness > 0.5:
        interpretation = "Moderately Right Skewed"

    elif skewness < -1:
        interpretation = "Highly Left Skewed"

    elif skewness < -0.5:
        interpretation = "Moderately Left Skewed"

    else:
        interpretation = "Approximately Symmetric"

    print(
        f"{column:<15}"
        f"Mean={df[column].mean():>10.2f}   "
        f"Median={df[column].median():>10.2f}   "
        f"{interpretation}"
    )

# =============================================================================
# 7. UNIVARIATE ANALYSIS — CATEGORICAL VARIABLES
# =============================================================================

print_separator("UNIVARIATE ANALYSIS — CATEGORICAL VARIABLES")

categorical_columns = [
    column
    for column in [
        "Segment",
        "Region",
        "Product_Category",
        "Order_Priority",
        "Season",
    ]
    if column in df.columns
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
)

axes = axes.flatten()

for axis, column in zip(axes, categorical_columns):

    counts = df[column].value_counts()

    axis.bar(
        counts.index.astype(str),
        counts.values,
    )

    axis.set_title(
        column,
        fontsize=12,
        fontweight="bold",
    )

    axis.set_xlabel("")
    axis.set_ylabel("Count")

    axis.tick_params(
        axis="x",
        rotation=40,
    )

    axis.grid(alpha=0.30)

    for index, value in enumerate(counts.values):

        axis.text(
            index,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

for axis in axes[len(categorical_columns):]:
    fig.delaxes(axis)

save_figure("06_categorical_distributions.png")

print("\nCategory Summary")

for column in categorical_columns:

    print(f"\n{column}")

    print(
        df[column]
        .value_counts()
        .rename("Count")
        .to_frame()
    )

# =============================================================================
# 8. BIVARIATE ANALYSIS
# =============================================================================

print_separator("BIVARIATE ANALYSIS")

# -----------------------------------------------------------------------------
# Product Category Performance
# -----------------------------------------------------------------------------

if {"Product_Category", "Sales"}.issubset(df.columns):

    category_sales = (
        df.groupby("Product_Category")["Sales"]
        .agg(
            Total_Sales="sum",
            Average_Sales="mean",
            Order_Count="count",
        )
        .sort_values(
            "Total_Sales",
            ascending=False,
        )
        .round(2)
    )

    print("\nSales by Product Category")

    print(category_sales)

    fig = px.bar(
        category_sales.reset_index(),
        x="Product_Category",
        y="Total_Sales",
        color="Total_Sales",
        title="Total Sales by Product Category",
        labels={
            "Product_Category": "Category",
            "Total_Sales": "Sales ($)",
        },
    )

    fig.write_html(
        FIGURES_DIR / "07_sales_by_category.html"
    )

# -----------------------------------------------------------------------------
# Regional Performance
# -----------------------------------------------------------------------------

if {"Region", "Sales"}.issubset(df.columns):

    region_sales = (
        df.groupby("Region")["Sales"]
        .agg(
            Total_Sales="sum",
            Average_Sales="mean",
            Order_Count="count",
        )
        .sort_values(
            "Total_Sales",
            ascending=False,
        )
        .round(2)
    )

    print("\nRegional Performance")

    print(region_sales)

    fig = px.pie(
        region_sales.reset_index(),
        values="Total_Sales",
        names="Region",
        hole=0.40,
        title="Revenue Distribution by Region",
    )

    fig.write_html(
        FIGURES_DIR / "08_sales_by_region.html"
    )

# -----------------------------------------------------------------------------
# Customer Segment Analysis
# -----------------------------------------------------------------------------

if {"Segment", "Sales"}.issubset(df.columns):

    segment_sales = (
        df.groupby("Segment")["Sales"]
        .agg(
            Total_Sales="sum",
            Average_Sales="mean",
            Order_Count="count",
        )
        .round(2)
    )

    print("\nCustomer Segment Performance")

    print(segment_sales)

# =============================================================================
# 9. CORRELATION ANALYSIS
# =============================================================================

print_separator("CORRELATION ANALYSIS")

correlation_columns = [
    column
    for column in [
        "Sales",
        "Quantity",
        "Profit",
        "Discount",
    ]
    if column in df.columns
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
    .round(2)
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    fmt=".2f",
)

plt.title(
    "Correlation Matrix",
    fontsize=14,
    fontweight="bold",
)

save_figure("09_correlation_matrix.png")

print("\nCorrelation Matrix")

print(correlation_matrix)

# =============================================================================
# 10. BUSINESS INSIGHTS
# =============================================================================

print_separator("KEY INSIGHTS")

if {"Product_Category", "Sales"}.issubset(df.columns):

    best_category = category_sales.index[0]

    print(
        f"✓ Highest revenue category : {best_category}"
    )

if {"Region", "Sales"}.issubset(df.columns):

    best_region = region_sales.index[0]

    print(
        f"✓ Highest revenue region   : {best_region}"
    )

if {"Segment", "Sales"}.issubset(df.columns):

    best_segment = (
        segment_sales["Total_Sales"]
        .idxmax()
    )

    print(
        f"✓ Best customer segment    : {best_segment}"
    )

print("\nStrongest Correlations")

corr_pairs = (
    correlation_matrix
    .where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1,
        ).astype(bool)
    )
    .stack()
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False,
    )
)

print(
    corr_pairs.rename("Correlation")
)

# =============================================================================
# 11. TIME SERIES ANALYSIS
# =============================================================================

print_separator("TIME SERIES ANALYSIS")

# -----------------------------------------------------------------------------
# Monthly Sales Trend
# -----------------------------------------------------------------------------

if {"Order_Date", "Sales"}.issubset(df.columns):

    monthly_sales = (
        df.assign(YearMonth=df["Order_Date"].dt.to_period("M").astype(str))
          .groupby("YearMonth", as_index=False)
          .agg(
              Total_Sales=("Sales", "sum"),
              Average_Order_Value=("Sales", "mean"),
              Order_Count=("Sales", "count"),
          )
    )

    print("\nMonthly Sales Summary")
    print(monthly_sales.tail(12))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_sales["YearMonth"],
            y=monthly_sales["Total_Sales"],
            mode="lines+markers",
            name="Sales",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title="Monthly Sales Trend",
        xaxis_title="Month",
        yaxis_title="Sales",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    fig.write_html(FIGURES_DIR / "10_monthly_sales_trend.html")

# -----------------------------------------------------------------------------
# Quarterly Sales Analysis
# -----------------------------------------------------------------------------

if {"Quarter", "Year", "Sales"}.issubset(df.columns):

    quarterly_sales = (
        df.groupby(["Year", "Quarter"], as_index=False)
          .agg(
              Total_Sales=("Sales", "sum"),
              Average_Order_Value=("Sales", "mean"),
              Orders=("Sales", "count"),
          )
    )

    print("\nQuarterly Performance")
    print(quarterly_sales)

    fig = px.bar(
        quarterly_sales,
        x="Quarter",
        y="Total_Sales",
        color="Year",
        barmode="group",
        title="Quarterly Sales Comparison",
        labels={
            "Quarter": "Quarter",
            "Total_Sales": "Sales",
        },
    )

    fig.write_html(FIGURES_DIR / "11_quarterly_sales.html")

# -----------------------------------------------------------------------------
# Sales by Day of Week
# -----------------------------------------------------------------------------

if {"Day_Name", "Sales"}.issubset(df.columns):

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_sales = (
        df.groupby("Day_Name")
          .agg(
              Total_Sales=("Sales", "sum"),
              Average_Sales=("Sales", "mean"),
              Orders=("Sales", "count"),
          )
          .reindex(weekday_order)
    )

    print("\nSales by Day of Week")
    print(weekday_sales)

    fig = px.bar(
        weekday_sales.reset_index(),
        x="Day_Name",
        y="Average_Sales",
        color="Average_Sales",
        title="Average Sales by Day",
    )

    fig.write_html(FIGURES_DIR / "12_sales_by_day.html")

# -----------------------------------------------------------------------------
# Weekend vs Weekday Performance
# -----------------------------------------------------------------------------

if {"Is_Weekend", "Sales"}.issubset(df.columns):

    weekend_sales = (
        df.groupby("Is_Weekend")
          .agg(
              Total_Sales=("Sales", "sum"),
              Average_Sales=("Sales", "mean"),
              Orders=("Sales", "count"),
          )
    )

    weekend_sales.index = ["Weekday", "Weekend"]

    print("\nWeekend vs Weekday Performance")
    print(weekend_sales)

    weekend_lift = (
        (
            weekend_sales.loc["Weekend", "Average_Sales"]
            /
            weekend_sales.loc["Weekday", "Average_Sales"]
        )
        - 1
    ) * 100

    print(f"\nWeekend Sales Lift : {weekend_lift:.2f}%")

# -----------------------------------------------------------------------------
# Seasonal Performance
# -----------------------------------------------------------------------------

if {"Season", "Sales"}.issubset(df.columns):

    season_order = [
        "Spring",
        "Summer",
        "Fall",
        "Winter",
    ]

    seasonal_sales = (
        df.groupby("Season")
          .agg(
              Total_Sales=("Sales", "sum"),
              Average_Sales=("Sales", "mean"),
              Orders=("Sales", "count"),
          )
          .reindex(season_order)
    )

    print("\nSeasonal Sales Performance")
    print(seasonal_sales)

    fig = px.bar(
        seasonal_sales.reset_index(),
        x="Season",
        y="Total_Sales",
        color="Total_Sales",
        title="Seasonal Revenue Distribution",
    )

    fig.write_html(FIGURES_DIR / "13_seasonal_sales.html")

# =============================================================================
# 12. TIME SERIES BUSINESS INSIGHTS
# =============================================================================

print_separator("TIME SERIES INSIGHTS")

if {"Order_Date", "Sales"}.issubset(df.columns):

    best_month = monthly_sales.loc[
        monthly_sales["Total_Sales"].idxmax(),
        "YearMonth",
    ]

    worst_month = monthly_sales.loc[
        monthly_sales["Total_Sales"].idxmin(),
        "YearMonth",
    ]

    print(f"Highest Revenue Month : {best_month}")
    print(f"Lowest Revenue Month  : {worst_month}")

if {"Quarter", "Sales"}.issubset(df.columns):

    quarterly_summary = (
        df.groupby("Quarter")["Sales"]
          .sum()
          .sort_values(ascending=False)
    )

    print(
        f"Best Performing Quarter : Q{quarterly_summary.index[0]}"
    )

if {"Day_Name", "Sales"}.issubset(df.columns):

    best_day = weekday_sales["Average_Sales"].idxmax()

    print(f"Best Sales Day         : {best_day}")

if {"Season", "Sales"}.issubset(df.columns):

    best_season = seasonal_sales["Total_Sales"].idxmax()

    print(f"Best Performing Season : {best_season}")

print(
    f"Weekend Sales Lift     : {weekend_lift:.2f}%"
)

# =============================================================================
# 13. CUSTOMER BEHAVIOR ANALYSIS
# =============================================================================

print_separator("CUSTOMER BEHAVIOR ANALYSIS")

# -----------------------------------------------------------------------------
# Customer Purchase Frequency
# -----------------------------------------------------------------------------

if "Customer_ID" in df.columns:

    customer_frequency = (
        df.groupby("Customer_ID")
          .size()
          .reset_index(name="Purchase_Count")
    )

    print("\nCustomer Purchase Statistics")
    print("-" * 40)
    print(f"Unique Customers           : {customer_frequency.shape[0]:,}")
    print(f"Average Purchases/Customer : {customer_frequency['Purchase_Count'].mean():.2f}")
    print(f"Median Purchases/Customer  : {customer_frequency['Purchase_Count'].median():.0f}")
    print(f"Maximum Purchases          : {customer_frequency['Purchase_Count'].max():.0f}")

    plt.figure(figsize=(12,6))

    plt.hist(
        customer_frequency["Purchase_Count"],
        bins=40,
        edgecolor="black",
        alpha=0.75,
    )

    plt.axvline(
        customer_frequency["Purchase_Count"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )

    plt.axvline(
        customer_frequency["Purchase_Count"].median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label="Median",
    )

    plt.title("Customer Purchase Frequency")
    plt.xlabel("Number of Purchases")
    plt.ylabel("Customers")
    plt.legend()

    save_figure("14_customer_purchase_frequency.png")

# -----------------------------------------------------------------------------
# Top Customers
# -----------------------------------------------------------------------------

if {"Customer_ID", "Sales"}.issubset(df.columns):

    top_customers = (
        df.groupby("Customer_ID")["Sales"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index(name="Revenue")
    )

    print("\nTop 10 Customers")
    print(top_customers)

    fig = px.bar(
        top_customers,
        x="Customer_ID",
        y="Revenue",
        color="Revenue",
        title="Top 10 Customers by Revenue",
    )

    fig.write_html(FIGURES_DIR / "15_top_customers.html")

# -----------------------------------------------------------------------------
# Customer Retention
# -----------------------------------------------------------------------------

if "Is_Repeat_Customer" in df.columns:

    retention = (
        df.groupby("Customer_ID")["Is_Repeat_Customer"]
          .max()
    )

    repeat_customers = retention.sum()
    total_customers = len(retention)

    retention_rate = (
        repeat_customers
        / total_customers
    ) * 100

    print("\nCustomer Retention")
    print("-" * 40)
    print(f"Repeat Customers : {repeat_customers:,}")
    print(f"One-time Customers : {total_customers-repeat_customers:,}")
    print(f"Retention Rate : {retention_rate:.2f}%")

# =============================================================================
# 14. PRODUCT PERFORMANCE ANALYSIS
# =============================================================================

print_separator("PRODUCT PERFORMANCE ANALYSIS")

# -----------------------------------------------------------------------------
# Top Performing Products
# -----------------------------------------------------------------------------

if {"Product_ID", "Sales"}.issubset(df.columns):

    product_performance = (
        df.groupby("Product_ID")
          .agg(
              Total_Revenue=("Sales","sum"),
              Total_Quantity=("Quantity","sum"),
              Order_Count=("Order_ID","count"),
          )
          .sort_values(
              "Total_Revenue",
              ascending=False,
          )
          .head(20)
    )

    print("\nTop 20 Products")
    print(product_performance)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=product_performance.index,
            y=product_performance["Total_Revenue"],
            name="Revenue",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=product_performance.index,
            y=product_performance["Total_Quantity"],
            mode="lines+markers",
            name="Quantity",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Top Products: Revenue vs Quantity",
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(title="Revenue"),
        yaxis2=dict(
            title="Quantity",
            overlaying="y",
            side="right",
        ),
    )

    fig.write_html(FIGURES_DIR / "16_top_products.html")

# -----------------------------------------------------------------------------
# Product Category Performance
# -----------------------------------------------------------------------------

if "Product_Category" in df.columns:

    category_matrix = (
        df.groupby("Product_Category")
          .agg(
              Total_Sales=("Sales","sum"),
              Average_Order_Value=("Sales","mean"),
              Orders=("Order_ID","count"),
              Customers=("Customer_ID","nunique"),
          )
    )

    category_matrix["Revenue_Share"] = (
        category_matrix["Total_Sales"]
        / category_matrix["Total_Sales"].sum()
        * 100
    ).round(2)

    print("\nCategory Performance")
    print(category_matrix.round(2))

    fig = px.scatter(
        category_matrix.reset_index(),
        x="Orders",
        y="Average_Order_Value",
        size="Total_Sales",
        color="Product_Category",
        hover_data=["Customers","Revenue_Share"],
        title="Category Performance Matrix",
    )

    fig.write_html(FIGURES_DIR / "17_category_performance.html")

# -----------------------------------------------------------------------------
# Pareto Analysis (80/20 Rule)
# -----------------------------------------------------------------------------

if {"Product_ID","Sales"}.issubset(df.columns):

    product_sales = (
        df.groupby("Product_ID")["Sales"]
          .sum()
          .sort_values(ascending=False)
    )

    cumulative_sales = (
        product_sales.cumsum()
        / product_sales.sum()
        * 100
    )

    top_products = (cumulative_sales <= 80).sum()

    pct_products = (
        top_products
        / len(product_sales)
        * 100
    )

    print("\nPareto Analysis")
    print("-" * 40)
    print(f"Products contributing to 80% Revenue : {top_products}")
    print(f"Percentage of Products : {pct_products:.2f}%")

# =============================================================================
# 15. CUSTOMER & PRODUCT INSIGHTS
# =============================================================================

print_separator("CUSTOMER & PRODUCT INSIGHTS")

if "Customer_ID" in df.columns:

    print(
        f"Average Purchases per Customer : "
        f"{customer_frequency['Purchase_Count'].mean():.2f}"
    )

    print(
        f"Customer Retention Rate : "
        f"{retention_rate:.2f}%"
    )

if {"Product_Category","Sales"}.issubset(df.columns):

    best_category = (
        category_matrix["Total_Sales"]
        .idxmax()
    )

    print(
        f"Highest Revenue Category : "
        f"{best_category}"
    )

if {"Product_ID","Sales"}.issubset(df.columns):

    best_product = product_sales.idxmax()

    print(
        f"Top Revenue Product : "
        f"{best_product}"
    )

    print(
        f"Top {pct_products:.1f}% of products contribute "
        f"approximately 80% of total revenue."
    )

# =============================================================================
# 16. ADVANCED BUSINESS INSIGHTS
# =============================================================================

print_separator("ADVANCED BUSINESS INSIGHTS")

# -----------------------------------------------------------------------------
# Sales Distribution
# -----------------------------------------------------------------------------

if "Sales" in df.columns:

    print("\nSales Distribution Summary")
    print("-" * 60)

    sales_summary = pd.Series({
        "Mean Sales": df["Sales"].mean(),
        "Median Sales": df["Sales"].median(),
        "Standard Deviation": df["Sales"].std(),
        "Minimum Sales": df["Sales"].min(),
        "Maximum Sales": df["Sales"].max(),
        "25th Percentile": df["Sales"].quantile(0.25),
        "75th Percentile": df["Sales"].quantile(0.75),
        "95th Percentile": df["Sales"].quantile(0.95),
        "99th Percentile": df["Sales"].quantile(0.99),
    }).round(2)

    print(sales_summary.to_frame("Value"))

# -----------------------------------------------------------------------------
# Discount Impact Analysis
# -----------------------------------------------------------------------------

if {"Discount", "Sales"}.issubset(df.columns):

    print("\nDiscount Impact Analysis")
    print("-" * 60)

    df["Discount_Group"] = pd.cut(
        df["Discount"],
        bins=[-0.01, 0, 0.10, 0.20, 1],
        labels=[
            "No Discount",
            "1–10%",
            "11–20%",
            "Above 20%",
        ],
    )

    discount_summary = (
        df.groupby("Discount_Group")
        .agg(
            Orders=("Sales", "count"),
            Average_Order_Value=("Sales", "mean"),
            Total_Revenue=("Sales", "sum"),
        )
        .round(2)
    )

    print(discount_summary)

# -----------------------------------------------------------------------------
# Delivery Performance
# -----------------------------------------------------------------------------

if "Delivery_Days" in df.columns:

    print("\nDelivery Performance")
    print("-" * 60)

    print(f"Average Delivery Time : {df['Delivery_Days'].mean():.2f} Days")
    print(f"Median Delivery Time  : {df['Delivery_Days'].median():.0f} Days")

    if "Delivery_Category" in df.columns:

        delivery_summary = (
            df["Delivery_Category"]
            .value_counts()
            .to_frame("Orders")
        )

        print(delivery_summary)

# =============================================================================
# 17. EXECUTIVE SUMMARY
# =============================================================================

print_separator("KEY BUSINESS FINDINGS")

findings = []

# -------------------------------------------------------------------------
# Highest Revenue Category
# -------------------------------------------------------------------------

if {"Product_Category", "Sales"}.issubset(df.columns):

    category_sales = (
        df.groupby("Product_Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    findings.append(
        f"• Highest Revenue Category: "
        f"{category_sales.idxmax()} "
        f"({category_sales.max()/category_sales.sum()*100:.1f}% of revenue)"
    )

# -------------------------------------------------------------------------
# Customer Retention
# -------------------------------------------------------------------------

if "Customer_ID" in df.columns:

    customer_orders = df.groupby("Customer_ID").size()

    repeat_rate = (
        (customer_orders > 1).sum()
        / len(customer_orders)
        * 100
    )

    findings.append(
        f"• Customer Retention Rate: {repeat_rate:.1f}%"
    )

# -------------------------------------------------------------------------
# Best Quarter
# -------------------------------------------------------------------------

if {"Quarter", "Sales"}.issubset(df.columns):

    quarter_sales = (
        df.groupby("Quarter")["Sales"]
        .sum()
    )

    findings.append(
        f"• Highest Sales Quarter: Q{quarter_sales.idxmax()}"
    )

# -------------------------------------------------------------------------
# Weekend Effect
# -------------------------------------------------------------------------

if {"Is_Weekend", "Sales"}.issubset(df.columns):

    weekday_avg = df.loc[df["Is_Weekend"] == 0, "Sales"].mean()
    weekend_avg = df.loc[df["Is_Weekend"] == 1, "Sales"].mean()

    weekend_lift = (
        (weekend_avg / weekday_avg) - 1
    ) * 100

    findings.append(
        f"• Weekend Sales Lift: {weekend_lift:.2f}%"
    )

# -------------------------------------------------------------------------
# Pareto Principle
# -------------------------------------------------------------------------

if {"Product_ID", "Sales"}.issubset(df.columns):

    revenue = (
        df.groupby("Product_ID")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    top_20 = int(len(revenue) * 0.20)

    revenue_share = (
        revenue.head(top_20).sum()
        / revenue.sum()
        * 100
    )

    findings.append(
        f"• Top 20% Products contribute {revenue_share:.1f}% of total revenue."
    )

# Print Findings

for item in findings:
    print(item)

# =============================================================================
# 18. EXPORT BUSINESS FINDINGS
# =============================================================================

report_text = f"""
RETAIL & MARKETING ANALYTICS PROJECT
EXPLORATORY DATA ANALYSIS REPORT
{'='*80}

Dataset Shape
-------------
Rows    : {df.shape[0]:,}
Columns : {df.shape[1]}

Business Findings
-----------------
{chr(10).join(findings)}

Generated On
------------
{pd.Timestamp.now().strftime('%d %B %Y %H:%M:%S')}
"""

report_path = REPORTS_DIR / "03_eda_summary_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_text)

print(f"\nReport Saved → {report_path}")

# =============================================================================
# 19. NOTEBOOK SUMMARY
# =============================================================================

print_separator("EDA COMPLETED")

print(
"""
✓ Dataset structure explored
✓ Numerical variables analyzed
✓ Categorical variables analyzed
✓ Correlation analysis completed
✓ Time series trends evaluated
✓ Customer purchasing behavior analyzed
✓ Product performance evaluated
✓ Discount & delivery analysis completed
✓ Business insights generated
✓ Summary report exported

Next Notebook:
Notebook 04 — Customer Segmentation & Advanced Analytics
"""
)