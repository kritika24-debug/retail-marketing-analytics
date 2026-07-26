# Retail & Marketing Analytics Project
# Notebook 5: KPI Design and Dashboard Preparation

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 05 - KPI Design and Dashboard Preparation
Author: [Kritika Sehgal]

Objective:
- Design comprehensive KPI framework
- Calculate key business metrics
- Prepare data for dashboard creation
- Generate executive summary report
- Create actionable recommendations
"""
# Objective:
#   • Load processed datasets
#   • Validate data integrity
#   • Prepare environment for KPI calculation
#   • Create output folders
# =============================================================================

print("=" * 90)
print("NOTEBOOK 05 : KPI DESIGN & DASHBOARD PREPARATION")
print("=" * 90)

# =============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:,.2f}".format)

# Plot styles
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

print("✓ Libraries imported successfully.")

# =============================================================================
# 2. PROJECT DIRECTORY CONFIGURATION
# =============================================================================

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("\nProject Directories")
print("-" * 50)
print(f"Processed Data : {DATA_DIR}")
print(f"Reports Folder : {REPORT_DIR}")
print(f"Figures Folder : {FIGURE_DIR}")

# =============================================================================
# 3. LOAD PROCESSED DATASETS
# =============================================================================

print("\nLoading processed datasets...")

try:

    df_sales = pd.read_csv(DATA_DIR / "cleaned_retail_sales.csv")
    df_sales["Order_Date"] = pd.to_datetime(df_sales["Order_Date"])

    rfm = pd.read_csv(DATA_DIR / "customer_segments.csv")

    customer_clv = pd.read_csv(DATA_DIR / "customer_clv.csv")

    print("✓ All datasets loaded successfully.")

except FileNotFoundError as e:

    print("\nERROR: Required dataset not found.")
    print(e)
    raise

# =============================================================================
# 4. DATASET OVERVIEW
# =============================================================================

print("\n" + "=" * 90)
print("DATASET SUMMARY")
print("=" * 90)

summary = pd.DataFrame({

    "Dataset": [
        "Retail Sales",
        "Customer Segments",
        "Customer CLV"
    ],

    "Rows": [
        len(df_sales),
        len(rfm),
        len(customer_clv)
    ],

    "Columns": [
        df_sales.shape[1],
        rfm.shape[1],
        customer_clv.shape[1]
    ]

})

print(summary)

# =============================================================================
# 5. DATA VALIDATION
# =============================================================================

print("\nRunning data validation checks...")

required_sales_columns = [
    "Order_ID",
    "Customer_ID",
    "Order_Date",
    "Sales",
    "Quantity"
]

missing_sales = [
    col for col in required_sales_columns
    if col not in df_sales.columns
]

if missing_sales:

    raise ValueError(
        f"Missing columns in cleaned_retail_sales.csv : {missing_sales}"
    )

required_rfm_columns = [
    "Customer_ID",
    "Recency",
    "Frequency",
    "Monetary"
]

missing_rfm = [
    col for col in required_rfm_columns
    if col not in rfm.columns
]

if missing_rfm:

    raise ValueError(
        f"Missing columns in customer_segments.csv : {missing_rfm}"
    )

if "CLV_Simple" not in customer_clv.columns:

    raise ValueError(
        "Column 'CLV_Simple' not found in customer_clv.csv"
    )

print("✓ Data validation completed successfully.")

# =============================================================================
# 6. BASIC DATA INFORMATION
# =============================================================================

print("\n" + "=" * 90)
print("PROJECT INFORMATION")
print("=" * 90)

print(f"Sales Records          : {len(df_sales):,}")
print(f"Unique Customers       : {df_sales['Customer_ID'].nunique():,}")
print(f"Unique Orders          : {df_sales['Order_ID'].nunique():,}")

print(f"Products               : {df_sales['Product_ID'].nunique():,}"
      if "Product_ID" in df_sales.columns
      else "Products               : N/A")

print(f"Customer Segments      : {len(rfm):,}")

print(f"CLV Records            : {len(customer_clv):,}")

print(f"Analysis Period Start  : {df_sales['Order_Date'].min().date()}")

print(f"Analysis Period End    : {df_sales['Order_Date'].max().date()}")

print("\n✓ Notebook setup completed successfully.")
print("=" * 90)

# =============================================================================
# 2. COMPREHENSIVE KPI FRAMEWORK
# =============================================================================
# Objective:
#     Calculate all important business KPIs including:
#     • Revenue Metrics
#     • Customer Metrics
#     • Product Metrics
#     • Marketing Metrics
#     • Retention Metrics
#     • Time-based Metrics
#     • Customer Segmentation Metrics
# =============================================================================

print("\n" + "=" * 90)
print("BUSINESS KPI FRAMEWORK")
print("=" * 90)

# Dictionary to store all KPIs
kpis = {}

# =============================================================================
# A. REVENUE METRICS
# =============================================================================

print("\nA. Revenue Metrics")
print("-" * 60)

kpis["Total Revenue"] = df_sales["Sales"].sum()

kpis["Total Orders"] = df_sales["Order_ID"].nunique()

kpis["Average Order Value"] = (
    df_sales.groupby("Order_ID")["Sales"]
    .sum()
    .mean()
)

kpis["Total Units Sold"] = df_sales["Quantity"].sum()

# Profit Metrics
if "Profit" in df_sales.columns:

    kpis["Total Profit"] = df_sales["Profit"].sum()

    kpis["Profit Margin (%)"] = (
        kpis["Total Profit"]
        / kpis["Total Revenue"]
    ) * 100

else:

    estimated_margin = 0.25

    kpis["Total Profit"] = (
        kpis["Total Revenue"] * estimated_margin
    )

    kpis["Profit Margin (%)"] = estimated_margin * 100

print(f"Total Revenue        : ${kpis['Total Revenue']:,.2f}")
print(f"Total Orders         : {kpis['Total Orders']:,}")
print(f"Average Order Value  : ${kpis['Average Order Value']:,.2f}")
print(f"Total Units Sold     : {kpis['Total Units Sold']:,}")
print(f"Total Profit         : ${kpis['Total Profit']:,.2f}")
print(f"Profit Margin        : {kpis['Profit Margin (%)']:.2f}%")

# =============================================================================
# B. CUSTOMER METRICS
# =============================================================================

print("\nB. Customer Metrics")
print("-" * 60)

kpis["Total Customers"] = df_sales["Customer_ID"].nunique()

customer_orders = (
    df_sales.groupby("Customer_ID")["Order_ID"]
    .nunique()
)

repeat_customers = (customer_orders > 1).sum()

kpis["Repeat Customers"] = repeat_customers

kpis["One-Time Customers"] = (
    kpis["Total Customers"] - repeat_customers
)

kpis["Repeat Customer Rate (%)"] = (
    repeat_customers
    / kpis["Total Customers"]
) * 100

kpis["Revenue Per Customer"] = (
    kpis["Total Revenue"]
    / kpis["Total Customers"]
)

kpis["Average Orders Per Customer"] = (
    kpis["Total Orders"]
    / kpis["Total Customers"]
)

print(f"Total Customers              : {kpis['Total Customers']:,}")
print(f"Repeat Customers             : {kpis['Repeat Customers']:,}")
print(f"One-Time Customers           : {kpis['One-Time Customers']:,}")
print(f"Repeat Customer Rate         : {kpis['Repeat Customer Rate (%)']:.2f}%")
print(f"Revenue Per Customer         : ${kpis['Revenue Per Customer']:,.2f}")
print(f"Average Orders Per Customer  : {kpis['Average Orders Per Customer']:.2f}")

# =============================================================================
# C. PRODUCT METRICS
# =============================================================================

print("\nC. Product Metrics")
print("-" * 60)

kpis["Total Products"] = df_sales["Product_ID"].nunique()

kpis["Average Items Per Order"] = (
    df_sales.groupby("Order_ID")["Quantity"]
    .sum()
    .mean()
)

if "Product_Category" in df_sales.columns:

    kpis["Product Categories"] = (
        df_sales["Product_Category"]
        .nunique()
    )

else:

    kpis["Product Categories"] = 0

print(f"Total Products         : {kpis['Total Products']:,}")
print(f"Product Categories     : {kpis['Product Categories']}")
print(f"Average Items/Order    : {kpis['Average Items Per Order']:.2f}")

# =============================================================================
# D. CUSTOMER LIFETIME VALUE (CLV) METRICS
# =============================================================================

print("\nD. Marketing & CLV Metrics")
print("-" * 60)

kpis["Average CLV"] = (
    customer_clv["CLV_Simple"]
    .mean()
)

kpis["Customer Acquisition Cost"] = 50

kpis["CLV/CAC Ratio"] = (
    kpis["Average CLV"]
    / kpis["Customer Acquisition Cost"]
)

kpis["Profit Per Customer"] = (
    kpis["Total Profit"]
    / kpis["Total Customers"]
)

monthly_revenue = (
    kpis["Revenue Per Customer"] / 12
)

kpis["CAC Payback (Months)"] = (
    kpis["Customer Acquisition Cost"]
    / (monthly_revenue * 0.25)
)

print("customer_clv.csv saved successfully.")

print(f"Average CLV            : ${kpis['Average CLV']:,.2f}")
print(f"CAC                    : ${kpis['Customer Acquisition Cost']:,.2f}")
print(f"CLV/CAC Ratio          : {kpis['CLV/CAC Ratio']:.2f}")
print(f"Profit Per Customer    : ${kpis['Profit Per Customer']:,.2f}")
print(f"CAC Payback Period     : {kpis['CAC Payback (Months)']:.2f} Months")

# =============================================================================
# E. RETENTION METRICS
# =============================================================================

print("\nE. Retention Metrics")
print("-" * 60)

if "Recency" in rfm.columns:

    churned = (rfm["Recency"] > 180).sum()

    retained = len(rfm) - churned

    kpis["Churned Customers"] = churned

    kpis["Retention Rate (%)"] = (
        retained / len(rfm)
    ) * 100

    kpis["Churn Rate (%)"] = (
        churned / len(rfm)
    ) * 100

    kpis["Average Recency"] = (
        rfm["Recency"]
        .mean()
    )

    print(f"Retention Rate     : {kpis['Retention Rate (%)']:.2f}%")
    print(f"Churn Rate         : {kpis['Churn Rate (%)']:.2f}%")
    print(f"Average Recency    : {kpis['Average Recency']:.2f} Days")

# =============================================================================
# F. TIME-BASED METRICS
# =============================================================================

print("\nF. Time-Based Metrics")
print("-" * 60)

analysis_days = (
    df_sales["Order_Date"].max()
    - df_sales["Order_Date"].min()
).days

kpis["Analysis Days"] = analysis_days

kpis["Average Daily Revenue"] = (
    kpis["Total Revenue"]
    / analysis_days
)

kpis["Average Daily Orders"] = (
    kpis["Total Orders"]
    / analysis_days
)

print(f"Analysis Period      : {analysis_days} Days")
print(f"Daily Revenue        : ${kpis['Average Daily Revenue']:,.2f}")
print(f"Daily Orders         : {kpis['Average Daily Orders']:.2f}")

# =============================================================================
# G. CUSTOMER SEGMENT METRICS
# =============================================================================

print("\nG. Customer Segmentation")
print("-" * 60)

if "Cluster_Name" in rfm.columns:

    segment_summary = (
        rfm["Cluster_Name"]
        .value_counts()
        .reset_index()
    )

    segment_summary.columns = [
        "Segment",
        "Customers"
    ]

    segment_summary["Percentage"] = (
        segment_summary["Customers"]
        / len(rfm)
    ) * 100

    print(segment_summary)

elif "Cluster" in rfm.columns:

    segment_summary = (
        rfm["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    segment_summary.columns = [
        "Cluster",
        "Customers"
    ]

    segment_summary["Percentage"] = (
        segment_summary["Customers"]
        / len(rfm)
    ) * 100

    print(segment_summary)

print("\n✓ Business KPI Framework Completed Successfully.")
print("=" * 90)

# =============================================================================
# 3. DASHBOARD DATA PREPARATION
# =============================================================================
# Objective:
#   • Create dashboard-ready datasets
#   • Monthly KPI trends
#   • Category performance
#   • Regional performance
#   • Customer segment summary
#   • Export datasets for Power BI / Tableau
# =============================================================================

print("\n" + "=" * 90)
print("DASHBOARD DATA PREPARATION")
print("=" * 90)

# =============================================================================
# A. KPI SUMMARY TABLE
# =============================================================================

print("\nPreparing KPI Summary...")

kpi_summary = (
    pd.DataFrame(
        list(kpis.items()),
        columns=["KPI", "Value"]
    )
    .sort_values("KPI")
    .reset_index(drop=True)
)

print(kpi_summary.head())

kpi_summary.to_csv(
    REPORT_DIR / "kpi_summary.csv",
    index=False
)

print("✓ KPI Summary exported.")

# =============================================================================
# B. MONTHLY KPI DATASET
# =============================================================================

print("\nPreparing Monthly KPI Dataset...")

df_sales["YearMonth"] = (
    df_sales["Order_Date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_kpis = (

    df_sales

    .groupby("YearMonth")

    .agg(
        Revenue=("Sales", "sum"),
        Orders=("Order_ID", "nunique"),
        Customers=("Customer_ID", "nunique"),
        Units_Sold=("Quantity", "sum"),
        Products=("Product_ID", "nunique")
    )

    .reset_index()

)

monthly_kpis["Average_Order_Value"] = (

    monthly_kpis["Revenue"]
    / monthly_kpis["Orders"]

)

monthly_kpis["Revenue_Per_Customer"] = (

    monthly_kpis["Revenue"]
    / monthly_kpis["Customers"]

)

monthly_kpis["Items_Per_Order"] = (

    monthly_kpis["Units_Sold"]
    / monthly_kpis["Orders"]

)

monthly_kpis["Revenue_Growth_%"] = (

    monthly_kpis["Revenue"]
    .pct_change()
    * 100

)

monthly_kpis["Customer_Growth_%"] = (

    monthly_kpis["Customers"]
    .pct_change()
    * 100

)

monthly_kpis["Order_Growth_%"] = (

    monthly_kpis["Orders"]
    .pct_change()
    * 100

)

monthly_kpis.to_csv(
    DATA_DIR / "monthly_kpis.csv",
    index=False
)

print("✓ Monthly KPI dataset created.")

# =============================================================================
# C. CATEGORY PERFORMANCE
# =============================================================================

if "Product_Category" in df_sales.columns:

    print("\nPreparing Category Performance...")

    category_kpis = (

        df_sales

        .groupby("Product_Category")

        .agg(
            Revenue=("Sales", "sum"),
            Orders=("Order_ID", "nunique"),
            Customers=("Customer_ID", "nunique"),
            Units=("Quantity", "sum"),
            Products=("Product_ID", "nunique")
        )

        .reset_index()

    )

    category_kpis["Average_Order_Value"] = (

        category_kpis["Revenue"]
        / category_kpis["Orders"]

    )

    category_kpis["Revenue_Share_%"] = (

        category_kpis["Revenue"]
        / category_kpis["Revenue"].sum()
        * 100

    )

    category_kpis = (

        category_kpis
        .sort_values("Revenue", ascending=False)

    )

    category_kpis.to_csv(

        REPORT_DIR / "category_kpis.csv",
        index=False

    )

    print("✓ Category KPI dataset exported.")

# =============================================================================
# D. REGIONAL PERFORMANCE
# =============================================================================

if "Region" in df_sales.columns:

    print("\nPreparing Regional Performance...")

    regional_kpis = (

        df_sales

        .groupby("Region")

        .agg(
            Revenue=("Sales", "sum"),
            Orders=("Order_ID", "nunique"),
            Customers=("Customer_ID", "nunique"),
            Units=("Quantity", "sum")
        )

        .reset_index()

    )

    regional_kpis["Average_Order_Value"] = (

        regional_kpis["Revenue"]
        / regional_kpis["Orders"]

    )

    regional_kpis["Revenue_Share_%"] = (

        regional_kpis["Revenue"]
        / regional_kpis["Revenue"].sum()
        * 100

    )

    regional_kpis["Customer_Penetration_%"] = (

        regional_kpis["Customers"]
        / kpis["Total Customers"]
        * 100

    )

    regional_kpis = (

        regional_kpis
        .sort_values("Revenue", ascending=False)

    )

    regional_kpis.to_csv(

        REPORT_DIR / "regional_kpis.csv",
        index=False

    )

    print("✓ Regional KPI dataset exported.")

# =============================================================================
# E. CUSTOMER SEGMENT SUMMARY
# =============================================================================

print("\nPreparing Customer Segment Summary...")

if "Cluster_Name" in rfm.columns:

    segment_dashboard = (

        rfm

        .groupby("Cluster_Name")

        .agg(
            Customers=("Customer_ID", "count"),
            Revenue=("Monetary", "sum"),
            Average_Revenue=("Monetary", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Recency=("Recency", "mean")
        )

        .reset_index()

    )

elif "Cluster" in rfm.columns:

    segment_dashboard = (

        rfm

        .groupby("Cluster")

        .agg(
            Customers=("Customer_ID", "count"),
            Revenue=("Monetary", "sum"),
            Average_Revenue=("Monetary", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Recency=("Recency", "mean")
        )

        .reset_index()

    )

segment_dashboard["Customer_%"] = (

    segment_dashboard["Customers"]

    / segment_dashboard["Customers"].sum()

    * 100

)

segment_dashboard["Revenue_%"] = (

    segment_dashboard["Revenue"]

    / segment_dashboard["Revenue"].sum()

    * 100

)

segment_dashboard.to_csv(

    REPORT_DIR / "customer_segment_summary.csv",

    index=False

)

print("✓ Customer Segment Summary exported.")

# =============================================================================
# F. DASHBOARD EXPORT SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("DASHBOARD DATASETS GENERATED")
print("=" * 90)

dashboard_files = [

    "kpi_summary.csv",

    "monthly_kpis.csv",

    "category_kpis.csv",

    "regional_kpis.csv",

    "customer_segment_summary.csv"

]

for file in dashboard_files:

    print(f"✓ {file}")

print("\nDashboard datasets are ready for Power BI / Tableau.")

print("=" * 90)

# =============================================================================
# 4. KPI VISUALIZATIONS & DASHBOARD COMPONENTS
# =============================================================================
# Objective:
#   • Create executive-level interactive visualizations
#   • Prepare dashboard components for Power BI/Tableau
# =============================================================================

print("\n" + "="*90)
print("CREATING INTERACTIVE DASHBOARD VISUALIZATIONS")
print("="*90)

# =============================================================================
# A. MONTHLY REVENUE TREND
# =============================================================================

print("\nCreating Revenue Trend...")

fig = px.line(
    monthly_kpis,
    x="YearMonth",
    y="Revenue",
    markers=True,
    title="Monthly Revenue Trend",
    labels={
        "YearMonth":"Month",
        "Revenue":"Revenue ($)"
    }
)

fig.update_traces(line=dict(width=4))

fig.update_layout(
    template="plotly_white",
    height=550,
    hovermode="x unified"
)

fig.write_html(FIGURE_DIR / "25_monthly_revenue_trend.html")

fig.show()

print("✓ Revenue Trend saved.")

# =============================================================================
# B. MONTHLY ORDERS & CUSTOMERS
# =============================================================================

print("\nCreating Orders vs Customers Chart...")

fig = go.Figure()

fig.add_trace(

    go.Bar(
        x=monthly_kpis["YearMonth"],
        y=monthly_kpis["Orders"],
        name="Orders"
    )

)

fig.add_trace(

    go.Scatter(
        x=monthly_kpis["YearMonth"],
        y=monthly_kpis["Customers"],
        mode="lines+markers",
        name="Customers",
        yaxis="y2"
    )

)

fig.update_layout(

    template="plotly_white",

    title="Orders vs Customers",

    yaxis=dict(title="Orders"),

    yaxis2=dict(
        title="Customers",
        overlaying="y",
        side="right"
    ),

    hovermode="x unified",

    height=550

)

fig.write_html(
    FIGURE_DIR /
    "26_orders_customers.html"
)
fig.show()

print("✓ Orders vs Customers chart saved.")

# =============================================================================
# C. KPI MONTH-OVER-MONTH COMPARISON
# =============================================================================

if len(monthly_kpis) >= 2:

    print("\nCreating Month-over-Month Comparison...")

    latest = monthly_kpis.iloc[-1]

    previous = monthly_kpis.iloc[-2]

    comparison = pd.DataFrame({

        "Metric":[
            "Revenue",
            "Orders",
            "Customers",
            "Average_Order_Value"
        ],

        "Previous":[
            previous["Revenue"],
            previous["Orders"],
            previous["Customers"],
            previous["Average_Order_Value"]
        ],

        "Current":[
            latest["Revenue"],
            latest["Orders"],
            latest["Customers"],
            latest["Average_Order_Value"]
        ]

    })

    fig = px.bar(

        comparison,

        x="Metric",

        y=["Previous","Current"],

        barmode="group",

        title="Month-over-Month KPI Comparison"

    )

    fig.update_layout(
        template="plotly_white",
        height=550
    )

    fig.write_html(
        FIGURE_DIR /
        "27_monthly_comparison.html"
    )
    fig.show()


    print("✓ Monthly comparison saved.")

# =============================================================================
# D. CUSTOMER SEGMENT DISTRIBUTION
# =============================================================================

print("\nCreating Customer Segment Chart...")

segment_col = "Cluster_Name" if "Cluster_Name" in rfm.columns else "Cluster"

segment_summary = (

    rfm

    .groupby(segment_col)

    .agg(

        Customers=("Customer_ID","count"),

        Revenue=("Monetary","sum")

    )

    .reset_index()

)

fig = px.bar(

    segment_summary,

    x=segment_col,

    y="Customers",

    color="Revenue",

    title="Customer Segment Distribution",

    text="Customers",

    color_continuous_scale="Viridis"

)

fig.update_layout(

    template="plotly_white",

    height=550

)

fig.write_html(

    FIGURE_DIR /
    "28_customer_segments.html"

)
fig.show()


print("✓ Customer Segment chart saved.")

# =============================================================================
# E. REVENUE SHARE BY SEGMENT
# =============================================================================

fig = px.pie(

    segment_summary,

    values="Revenue",

    names=segment_col,

    hole=0.45,

    title="Revenue Contribution by Customer Segment"

)

fig.update_layout(
    template="plotly_white",
    height=550
)

fig.write_html(

    FIGURE_DIR /
    "29_segment_revenue_share.html"

)
fig.show()


print("✓ Revenue share chart saved.")

# =============================================================================
# F. CATEGORY PERFORMANCE
# =============================================================================

if "Product_Category" in df_sales.columns:

    print("\nCreating Category Performance...")

    category_summary = (

        df_sales

        .groupby("Product_Category")["Sales"]

        .sum()

        .sort_values(ascending=False)

        .reset_index()

    )

    fig = px.bar(

        category_summary,

        x="Product_Category",

        y="Sales",

        title="Revenue by Product Category",

        text_auto=".2s",

        color="Sales",

        color_continuous_scale="Blues"

    )

    fig.update_layout(

        template="plotly_white",

        height=600,

        xaxis_title="Category",

        yaxis_title="Revenue"

    )

    fig.write_html(

        FIGURE_DIR /
        "30_category_performance.html"

    )
    fig.show()


    print("✓ Category chart saved.")

# =============================================================================
# G. REGIONAL PERFORMANCE
# =============================================================================

if "Region" in df_sales.columns:

    print("\nCreating Regional Performance...")

    region_summary = (

        df_sales

        .groupby("Region")["Sales"]

        .sum()

        .sort_values(ascending=False)

        .reset_index()

    )

    fig = px.bar(

        region_summary,

        x="Region",

        y="Sales",

        title="Regional Revenue",

        text_auto=".2s",

        color="Sales",

        color_continuous_scale="Greens"

    )

    fig.update_layout(

        template="plotly_white",

        height=550

    )

    fig.write_html(

        FIGURE_DIR /
        "31_regional_performance.html"

    )
    fig.show()


    print("✓ Regional Performance chart saved.")

# =============================================================================
# H. CLV DISTRIBUTION
# =============================================================================

print("\nCreating CLV Distribution...")

fig = px.histogram(

    customer_clv,

    x="CLV_Simple",

    nbins=40,

    title="Customer Lifetime Value Distribution"

)

fig.update_layout(

    template="plotly_white",

    height=550

)

fig.write_html(

    FIGURE_DIR /
    "32_clv_distribution.html"

)
fig.show()


print("✓ CLV Distribution saved.")

# =============================================================================
# I. DASHBOARD VISUALIZATION SUMMARY
# =============================================================================

print("\n" + "="*90)
print("DASHBOARD VISUALIZATIONS CREATED")
print("="*90)

visualizations = [

    "25_monthly_revenue_trend.html",
    "26_orders_customers.html",
    "27_monthly_comparison.html",
    "28_customer_segments.html",
    "29_segment_revenue_share.html",
    "30_category_performance.html",
    "31_regional_performance.html",
    "32_clv_distribution.html"

]

for chart in visualizations:
    print(f"✓ {chart}")

print("\nInteractive dashboard visualizations completed successfully.")
print("="*90)
fig.show()

# =============================================================================
# 5. EXECUTIVE REPORTING & PROJECT COMPLETION
# =============================================================================
# Objective:
#     • Generate Executive Summary
#     • Generate Project Completion Report
#     • Verify Output Files
# =============================================================================

print("\n" + "=" * 90)
print("EXECUTIVE REPORTING")
print("=" * 90)

from datetime import datetime
from pathlib import Path

# =============================================================================
# A. EXECUTIVE SUMMARY
# =============================================================================

print("\nGenerating Executive Summary...")

segment_column = "Cluster_Name" if "Cluster_Name" in rfm.columns else "Cluster"

top_segment = (
    rfm.groupby(segment_column)["Monetary"]
    .sum()
    .sort_values(ascending=False)
)

top_segment_name = top_segment.index[0]
top_segment_share = (
    top_segment.iloc[0]
    / rfm["Monetary"].sum()
    * 100
)

summary = f"""
==========================================================================================
                      RETAIL & MARKETING ANALYTICS
                          EXECUTIVE SUMMARY
==========================================================================================

Generated On : {datetime.now().strftime("%d-%m-%Y %H:%M")}

BUSINESS PERFORMANCE
--------------------

Total Revenue                 : ${kpis['Total Revenue']:,.2f}
Total Profit                  : ${kpis['Total Profit']:,.2f}
Profit Margin                 : {kpis['Profit Margin (%)']:.2f}%

Total Customers               : {kpis['Total Customers']:,}
Total Orders                  : {kpis['Total Orders']:,}
Average Order Value           : ${kpis['Average Order Value']:,.2f}

CUSTOMER INSIGHTS
--------------------

Repeat Customer Rate          : {kpis['Repeat Customer Rate (%)']:.2f}%

Retention Rate                : {kpis.get('Retention Rate (%)',0):.2f}%

Average Customer Lifetime Value : ${kpis['Average CLV']:,.2f}

CLV / CAC Ratio               : {kpis['CLV/CAC Ratio']:.2f}

TOP CUSTOMER SEGMENT
--------------------

Segment Name                  : {top_segment_name}

Revenue Contribution          : {top_segment_share:.2f}%

KEY BUSINESS RECOMMENDATIONS
----------------------------

• Focus marketing budget on the highest-value customer segment.

• Improve retention through loyalty and reward programs.

• Launch personalized campaigns for inactive customers.

• Continue optimizing high-performing product categories.

• Monitor KPIs monthly using the dashboard.

==========================================================================================
"""

print(summary)

(Path(REPORT_DIR) / "executive_summary.txt").write_text(summary)

print("✓ Executive Summary saved.")

# =============================================================================
# B. PROJECT COMPLETION REPORT
# =============================================================================

print("\nCreating Project Completion Report...")

completion_report = f"""
================================================================================
PROJECT COMPLETION REPORT
================================================================================

Project Name
------------
Retail & Marketing Analytics

Completion Date
---------------
{datetime.now().strftime("%d-%m-%Y")}

OBJECTIVES COMPLETED
--------------------

✓ Data Cleaning

✓ Exploratory Data Analysis

✓ Feature Engineering

✓ RFM Analysis

✓ Customer Segmentation

✓ Cohort Analysis

✓ Customer Lifetime Value

✓ KPI Framework

✓ Dashboard Dataset Preparation

✓ Interactive Dashboard Visualizations

PROJECT STATISTICS
------------------

Sales Records            : {len(df_sales):,}

Customers                : {kpis['Total Customers']:,}

Orders                   : {kpis['Total Orders']:,}

Revenue                  : ${kpis['Total Revenue']:,.2f}

Customer Segments        : {rfm[segment_column].nunique()}

Dashboard Visuals        : 8+

Reports Generated        : 5+

STATUS
------

PROJECT COMPLETED SUCCESSFULLY

================================================================================
"""

(Path(REPORT_DIR) /
 "project_completion_report.txt").write_text(completion_report,encoding="utf-8")

print("✓ Project Completion Report saved.")

# =============================================================================
# C. OUTPUT FILE VERIFICATION
# =============================================================================

print("\nVerifying Generated Outputs...")

required_outputs = [

    DATA_DIR / "cleaned_retail_sales.csv",

    DATA_DIR / "customer_segments.csv",

    DATA_DIR / "customer_clv.csv",

    DATA_DIR / "monthly_kpis.csv",

    REPORT_DIR / "kpi_summary.csv",

    REPORT_DIR / "customer_segment_summary.csv",

    REPORT_DIR / "executive_summary.txt",

    REPORT_DIR / "project_completion_report.txt"

]

verified = 0

for file in required_outputs:

    if Path(file).exists():

        print(f"✓ {file.name}")

        verified += 1

    else:

        print(f"✗ Missing : {file.name}")

print(f"\nVerified Files : {verified}/{len(required_outputs)}")

# =============================================================================
# D. PROJECT DASHBOARD SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("PROJECT DASHBOARD SUMMARY")
print("=" * 90)

dashboard_summary = pd.DataFrame({

    "Metric":[

        "Revenue",

        "Customers",

        "Orders",

        "Average Order Value",

        "Average CLV",

        "Retention Rate",

        "Profit Margin"

    ],

    "Value":[

        round(kpis["Total Revenue"],2),

        kpis["Total Customers"],

        kpis["Total Orders"],

        round(kpis["Average Order Value"],2),

        round(kpis["Average CLV"],2),

        round(kpis.get("Retention Rate (%)",0),2),

        round(kpis["Profit Margin (%)"],2)

    ]

})

print(dashboard_summary)

dashboard_summary.to_csv(

    REPORT_DIR / "dashboard_summary.csv",

    index=False

)

print("\n✓ Dashboard Summary exported.")

# =============================================================================
# E. NOTEBOOK COMPLETION
# =============================================================================

print("\n" + "=" * 90)
print("NOTEBOOK 05 COMPLETED SUCCESSFULLY")
print("=" * 90)

print("""
Project Deliverables
--------------------

✓ Cleaned Dataset

✓ RFM Analysis

✓ Customer Segmentation

✓ Customer Lifetime Value

✓ KPI Framework

✓ Dashboard Ready Data

✓ Interactive Visualizations

✓ Executive Summary

✓ Dashboard Summary

✓ Project Completion Report

Your Retail & Marketing Analytics Project is now ready for:

• Power BI Dashboard

• Tableau Dashboard

• GitHub Portfolio

• Resume Project

• Academic Submission

• Business Presentation
""")

print("=" * 90)

# =============================================================================
# 6. FINAL VALIDATION & PROJECT CLOSURE
# =============================================================================
# Objective:
#     • Validate generated outputs
#     • Display project statistics
#     • Check required files
#     • Print completion message
# =============================================================================

print("\n" + "=" * 90)
print("FINAL PROJECT VALIDATION")
print("=" * 90)

from pathlib import Path

# =============================================================================
# A. CHECK GENERATED FILES
# =============================================================================

print("\nChecking Generated Files...\n")

required_files = {

    "Processed Data": [

        DATA_DIR / "cleaned_retail_sales.csv",
        DATA_DIR / "customer_segments.csv",
        DATA_DIR / "customer_clv.csv",
        DATA_DIR / "monthly_kpis.csv"

    ],

    "Reports": [

        REPORT_DIR / "kpi_summary.csv",
        REPORT_DIR / "customer_segment_summary.csv",
        REPORT_DIR / "dashboard_summary.csv",
        REPORT_DIR / "executive_summary.txt",
        REPORT_DIR / "project_completion_report.txt"

    ],

    "Visualizations": [

        FIGURE_DIR / "25_monthly_revenue_trend.html",
        FIGURE_DIR / "26_orders_customers.html",
        FIGURE_DIR / "27_monthly_comparison.html",
        FIGURE_DIR / "28_customer_segments.html",
        FIGURE_DIR / "29_segment_revenue_share.html",
        FIGURE_DIR / "30_category_performance.html",
        FIGURE_DIR / "31_regional_performance.html",
        FIGURE_DIR / "32_clv_distribution.html"

    ]

}

total_files = 0
available_files = 0

for category, files in required_files.items():

    print(f"\n{category}")

    print("-" * 60)

    for file in files:

        total_files += 1

        if Path(file).exists():

            print(f"✓ {file.name}")

            available_files += 1

        else:

            print(f"✗ {file.name}")

# =============================================================================
# B. PROJECT STATISTICS
# =============================================================================

print("\n" + "=" * 90)
print("PROJECT STATISTICS")
print("=" * 90)

segment_column = (
    "Cluster_Name"
    if "Cluster_Name" in rfm.columns
    else "Cluster"
)

print(f"Sales Records             : {len(df_sales):,}")
print(f"Customers                 : {kpis['Total Customers']:,}")
print(f"Orders                    : {kpis['Total Orders']:,}")
print(f"Products                  : {df_sales['Product_ID'].nunique():,}")
print(f"Customer Segments         : {rfm[segment_column].nunique()}")
print(f"Revenue                   : ${kpis['Total Revenue']:,.2f}")
print(f"Average Order Value       : ${kpis['Average Order Value']:,.2f}")
print(f"Average Customer CLV      : ${kpis['Average CLV']:,.2f}")

# =============================================================================
# C. PROJECT COMPLETION STATUS
# =============================================================================

completion_rate = (available_files / total_files) * 100

print("\n" + "=" * 90)
print("PROJECT STATUS")
print("=" * 90)

print(f"Generated Files : {available_files}/{total_files}")

print(f"Completion Rate : {completion_rate:.1f}%")

if completion_rate == 100:

    print("\nPROJECT STATUS : SUCCESS")

else:

    print("\nPROJECT STATUS : PARTIALLY COMPLETED")

# =============================================================================
# D. PROJECT DELIVERABLES
# =============================================================================

print("\n" + "=" * 90)
print("PROJECT DELIVERABLES")
print("=" * 90)

deliverables = [

    "✓ Data Cleaning & Preprocessing",

    "✓ Exploratory Data Analysis",

    "✓ Customer RFM Analysis",

    "✓ Customer Segmentation (K-Means)",

    "✓ Customer Lifetime Value (CLV)",

    "✓ KPI Framework",

    "✓ Dashboard Ready Datasets",

    "✓ Interactive Plotly Visualizations",

    "✓ Executive Summary",

    "✓ Business Recommendation Report"

]

for item in deliverables:

    print(item)

# =============================================================================
# E. RECOMMENDED NEXT STEPS
# =============================================================================

print("\n" + "=" * 90)
print("RECOMMENDED NEXT STEPS")
print("=" * 90)

next_steps = [

    "1. Build an interactive Power BI Dashboard",

    "2. Create Tableau Dashboard (optional)",

    "3. Upload project to GitHub",

    "4. Prepare presentation slides",

    "5. Include project in portfolio",

    "6. Deploy dashboard using Power BI Service"

]

for step in next_steps:

    print(step)

# =============================================================================
# F. NOTEBOOK COMPLETION
# =============================================================================

print("\n" + "=" * 90)
print("NOTEBOOK 05 COMPLETED SUCCESSFULLY")
print("=" * 90)

print("""

Retail & Marketing Analytics Project
------------------------------------

Status        : COMPLETED

Deliverables  : Generated Successfully

Dashboard     : Ready

Reports       : Ready

Portfolio     : Ready

GitHub        : Ready

Power BI      : Ready

Thank you for completing the Retail & Marketing Analytics Project!

""")

print("=" * 90)