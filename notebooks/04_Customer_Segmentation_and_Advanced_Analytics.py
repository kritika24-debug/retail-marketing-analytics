# Retail & Marketing Analytics Project
# Notebook 4: Customer Segmentation and Advanced Analytics

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 04 - Customer Segmentation and Advanced Analytics
Author: [Kritika Sehgal]

Objective:
- RFM Analysis (Recency, Frequency, Monetary)
- K-Means Clustering for customer segmentation
- Cluster profiling and characterization
- Market Basket Analysis
- Cohort Analysis
- Customer Lifetime Value (CLV) calculation
"""

# ==============================================================================
# 1. IMPORT LIBRARIES
# ==============================================================================

import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==============================================================================
# 2. OPTIONAL PACKAGE (MARKET BASKET ANALYSIS)
# ==============================================================================

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    MBA_AVAILABLE = True

except ImportError:
    MBA_AVAILABLE = False
    print("⚠ mlxtend not installed.")
    print("Install using:")
    print("pip install mlxtend")

# ==============================================================================
# 3. PROJECT PATHS
# ==============================================================================

BASE_DIR = Path(".")

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

for directory in [
    PROCESSED_DIR,
    FIGURE_DIR,
    REPORT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 4. PLOTTING CONFIGURATION
# ==============================================================================

plt.style.use("ggplot")

sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="Set2"
)

plt.rcParams.update({

    "figure.figsize": (12,6),
    "figure.dpi":120,

    "axes.titlesize":16,
    "axes.labelsize":12,

    "xtick.labelsize":10,
    "ytick.labelsize":10,

    "legend.fontsize":10

})

COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#059669"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#DC2626"

# ==============================================================================
# 5. HELPER FUNCTIONS
# ==============================================================================

def section(title):
    """Print notebook section."""

    print("\n" + "="*90)
    print(title.upper())
    print("="*90)


def save_figure(filename):

    filepath = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Saved -> {filepath}")

    plt.close()


def save_plotly(fig, filename):

    filepath = FIGURE_DIR / filename

    fig.write_html(filepath)

    print(f"✓ Saved -> {filepath}")


def print_shape(dataframe):

    print(f"Rows    : {dataframe.shape[0]:,}")
    print(f"Columns : {dataframe.shape[1]:,}")


def missing_summary(dataframe):

    missing = dataframe.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        print("✓ No missing values detected.")
    else:
        print(missing.sort_values(ascending=False))


# ==============================================================================
# 6. LOAD CLEANED DATA
# ==============================================================================

section("Loading Cleaned Dataset")

DATA_FILE = PROCESSED_DIR / "cleaned_retail_sales.csv"

df = pd.read_csv(DATA_FILE)

# ------------------------------------------------------------------------------
# Convert datetime columns
# ------------------------------------------------------------------------------

date_columns = [
    "Order_Date",
    "Ship_Date"
]

for column in date_columns:

    if column in df.columns:
        df[column] = pd.to_datetime(df[column])

print("✓ Dataset loaded successfully.\n")

print_shape(df)

print("\nDataset Memory Usage")

memory = df.memory_usage(deep=True).sum() / 1024**2

print(f"{memory:.2f} MB")

print("\nAnalysis Period")

print(
    f"{df['Order_Date'].min().date()}  -->  {df['Order_Date'].max().date()}"
)

print("\nUnique Customers :", df["Customer_ID"].nunique())

print("Unique Orders    :", df["Order_ID"].nunique())

print("Unique Products  :", df["Product_ID"].nunique())

# ==============================================================================
# 7. DATA QUALITY CHECK
# ==============================================================================

section("Dataset Validation")

missing_summary(df)

duplicate_orders = df.duplicated().sum()

print(f"\nDuplicate Rows : {duplicate_orders:,}")

print("\nNumerical Summary")

print(
    df.select_dtypes(include=np.number)
      .describe()
      .round(2)
)

print("\nDataset Preview")

print(df.head())

print("\n✓ Dataset validation completed.")

section("RFM ANALYSIS")

# ------------------------------------------------------------------------------
# Analysis Date
# ------------------------------------------------------------------------------

analysis_date = df["Order_Date"].max() + timedelta(days=1)

print(f"Analysis Date : {analysis_date.date()}")

# ------------------------------------------------------------------------------
# Customer Level Aggregation
# ------------------------------------------------------------------------------

rfm = (
    df.groupby("Customer_ID")
      .agg(
          Recency=("Order_Date",
                   lambda x: (analysis_date - x.max()).days),

          Frequency=("Order_ID", "nunique"),

          Monetary=("Sales", "sum")
      )
      .reset_index()
)

print(f"\nTotal Customers : {len(rfm):,}")

print("\nRFM Summary Statistics")

print(
    rfm[
        ["Recency", "Frequency", "Monetary"]
    ].describe().round(2)
)

# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------

def create_score(series, reverse=False):
    """
    Create RFM score using quintiles.

    reverse=True
    Smaller values receive higher score.
    """

    rank = series.rank(method="first")

    if reverse:
        score = pd.qcut(
            rank,
            5,
            labels=[5,4,3,2,1],
            duplicates="drop"
        )
    else:
        score = pd.qcut(
            rank,
            5,
            labels=[1,2,3,4,5],
            duplicates="drop"
        )

    return score.astype(int)

# ------------------------------------------------------------------------------
# Generate Scores
# ------------------------------------------------------------------------------

print("\nCreating RFM Scores...")

rfm["R_Score"] = create_score(
    rfm["Recency"],
    reverse=True
)

rfm["F_Score"] = create_score(
    rfm["Frequency"]
)

rfm["M_Score"] = create_score(
    rfm["Monetary"]
)

rfm["RFM_Code"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

rfm["RFM_Score"] = (
    rfm["R_Score"]
    + rfm["F_Score"]
    + rfm["M_Score"]
)

print("✓ RFM Scores Created")

# ------------------------------------------------------------------------------
# Segment Assignment
# ------------------------------------------------------------------------------

def assign_segment(row):

    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r >= 4 and f >= 2 and m >= 3:
        return "Potential Loyalists"

    elif r >= 4 and f <= 2:
        return "New Customers"

    elif r == 3 and f >= 3:
        return "Need Attention"

    elif r <= 2 and f >= 4:
        return "At Risk"

    elif r <= 2 and f <= 2 and m >= 3:
        return "About to Sleep"

    elif r == 1 and f <= 2:
        return "Lost Customers"

    else:
        return "Others"

rfm["Customer_Segment"] = rfm.apply(
    assign_segment,
    axis=1
)

# ------------------------------------------------------------------------------
# Segment Distribution
# ------------------------------------------------------------------------------

section("Customer Segment Distribution")

segment_summary = (
    rfm["Customer_Segment"]
    .value_counts()
    .rename_axis("Segment")
    .reset_index(name="Customers")
)

segment_summary["Percentage"] = (
    segment_summary["Customers"]
    / segment_summary["Customers"].sum()
    *100
).round(2)

print(segment_summary)

# ------------------------------------------------------------------------------
# Segment Performance
# ------------------------------------------------------------------------------

rfm_summary = (
    rfm.groupby("Customer_Segment")
       .agg(
           Customers=("Customer_ID","count"),
           Avg_Recency=("Recency","mean"),
           Avg_Frequency=("Frequency","mean"),
           Avg_Monetary=("Monetary","mean"),
           Total_Revenue=("Monetary","sum")
       )
       .round(2)
       .sort_values(
           "Total_Revenue",
           ascending=False
       )
)

rfm_summary["Revenue_%"] = (
    rfm_summary["Total_Revenue"]
    / rfm_summary["Total_Revenue"].sum()
    *100
).round(2)

section("RFM Segment Performance")

print(rfm_summary)

# ------------------------------------------------------------------------------
# Save Outputs
# ------------------------------------------------------------------------------

rfm.to_csv(
    PROCESSED_DIR / "rfm_analysis.csv",
    index=False
)

rfm_summary.to_csv(
    REPORT_DIR / "rfm_segment_summary.csv"
)

print("\n✓ RFM analysis exported successfully.")


section("RFM VISUALIZATION")

# ==============================================================================
# 1. CUSTOMER SEGMENT DISTRIBUTION
# ==============================================================================

plt.figure(figsize=(12,6))

segment_counts = (
    rfm["Customer_Segment"]
    .value_counts()
    .sort_values(ascending=False)
)

bars = plt.bar(
    segment_counts.index,
    segment_counts.values,
    color=sns.color_palette("Set2", len(segment_counts)),
    edgecolor="black"
)

plt.title(
    "Customer Distribution by RFM Segment",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.xticks(rotation=20)

for bar in bars:

    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{int(bar.get_height())}",
        ha="center",
        va="bottom",
        fontsize=10
    )

save_figure("17_rfm_segment_distribution.png")

# ==============================================================================
# 2. INTERACTIVE PIE CHART
# ==============================================================================

fig = px.pie(

    segment_summary,

    names="Segment",

    values="Customers",

    hole=0.45,

    title="Customer Segmentation using RFM Analysis",

    color_discrete_sequence=px.colors.qualitative.Set3

)

save_plotly(
    fig,
    "17_rfm_segments.html"
)

# ==============================================================================
# 3. REVENUE CONTRIBUTION
# ==============================================================================

section("Revenue Contribution by Segment")

revenue_df = (

    rfm.groupby("Customer_Segment")["Monetary"]

       .sum()

       .sort_values(ascending=False)

       .reset_index()

)

revenue_df.columns = [

    "Segment",

    "Revenue"

]

revenue_df["Revenue %"] = (

    revenue_df["Revenue"]

    / revenue_df["Revenue"].sum()

    *100

).round(2)

print(revenue_df)

plt.figure(figsize=(12,6))

bars = plt.bar(

    revenue_df["Segment"],

    revenue_df["Revenue"],

    color="steelblue",

    edgecolor="black"

)

plt.title(

    "Revenue Contribution by Customer Segment",

    fontsize=16,

    fontweight="bold"

)

plt.xlabel("Customer Segment")

plt.ylabel("Revenue")

plt.xticks(rotation=20)

for bar, pct in zip(bars, revenue_df["Revenue %"]):

    plt.text(

        bar.get_x()+bar.get_width()/2,

        bar.get_height(),

        f"{pct:.1f}%",

        ha="center",

        fontsize=10

    )

save_figure("18_segment_revenue.png")

# ==============================================================================
# 4. RFM HEATMAP
# ==============================================================================

section("Average RFM Scores")

heatmap_df = (

    rfm.groupby("Customer_Segment")[

        ["Recency","Frequency","Monetary"]

    ]

    .mean()

    .round(1)

)

plt.figure(figsize=(10,6))

sns.heatmap(

    heatmap_df,

    annot=True,

    cmap="YlGnBu",

    linewidths=1,

    fmt=".1f"

)

plt.title(

    "Average RFM Metrics by Segment",

    fontsize=15,

    fontweight="bold"

)

save_figure("19_rfm_heatmap.png")

# ==============================================================================
# 5. SEGMENT BUSINESS RECOMMENDATIONS
# ==============================================================================

section("BUSINESS RECOMMENDATIONS")

recommendations = {

    "Champions":[

        "Reward loyalty with VIP programs",

        "Offer early product access",

        "Encourage referrals"

    ],

    "Loyal Customers":[

        "Cross-sell premium products",

        "Introduce loyalty rewards",

        "Personalized email campaigns"

    ],

    "Potential Loyalists":[

        "Offer limited-time discounts",

        "Recommend related products",

        "Increase engagement"

    ],

    "New Customers":[

        "Welcome email series",

        "First repeat purchase coupon",

        "Introduce best-selling products"

    ],

    "Need Attention":[

        "Re-engagement campaigns",

        "Personalized promotions"

    ],

    "At Risk":[

        "Win-back offers",

        "Exclusive discount coupons",

        "Customer feedback survey"

    ],

    "About to Sleep":[

        "Reminder emails",

        "Flash sales",

        "Seasonal promotions"

    ],

    "Lost Customers":[

        "Aggressive win-back campaign",

        "Special incentives",

        "Exit survey"

    ],

    "Others":[

        "General marketing campaigns"

    ]

}

for segment in segment_summary["Segment"]:

    print("\n"+"="*70)

    print(segment.upper())

    print("="*70)

    for item in recommendations.get(segment,["General Promotion"]):

        print(f"• {item}")

# ==============================================================================
# 6. EXECUTIVE SUMMARY
# ==============================================================================

section("EXECUTIVE SUMMARY")

largest_segment = segment_summary.iloc[0]

highest_revenue = revenue_df.iloc[0]

print(f"Largest Segment        : {largest_segment['Segment']}")
print(f"Customers             : {largest_segment['Customers']:,}")

print()

print(f"Highest Revenue Group : {highest_revenue['Segment']}")
print(f"Revenue Share         : {highest_revenue['Revenue %']:.2f}%")

print()

print(f"Total Customers       : {len(rfm):,}")

print(f"Average Recency       : {rfm['Recency'].mean():.1f} Days")

print(f"Average Frequency     : {rfm['Frequency'].mean():.2f} Orders")

print(f"Average Monetary      : ${rfm['Monetary'].mean():,.2f}")

# ==============================================================================
# 7. SAVE FINAL DATA
# ==============================================================================

rfm.to_csv(

    PROCESSED_DIR / "customer_segments.csv",

    index=False

)

print("\n✓ customer_segments.csv saved")

print("✓ RFM Analysis Completed Successfully")


section("K-MEANS CLUSTERING")

# ==============================================================================
# 1. PREPARE CLUSTERING DATA
# ==============================================================================

cluster_features = [
    "Recency",
    "Frequency",
    "Monetary"
]

print("\nSelected Features")
print("-" * 60)

for feature in cluster_features:
    print(f"• {feature}")

X = rfm[cluster_features].copy()

print(f"\nDataset Shape : {X.shape}")

# ==============================================================================
# 2. CHECK MISSING VALUES
# ==============================================================================

print("\nChecking Missing Values...")

missing = X.isnull().sum()

if missing.sum() == 0:
    print("✓ No missing values detected.")
else:
    print(missing)

    X = X.fillna(X.median())

    print("✓ Missing values replaced using median.")

# ==============================================================================
# 3. DESCRIPTIVE STATISTICS
# ==============================================================================

section("FEATURE SUMMARY")

print(X.describe().round(2))

# ==============================================================================
# 4. STANDARDIZE FEATURES
# ==============================================================================

print("\nStandardizing Features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=cluster_features
)

print("✓ Feature scaling completed.")

print("\nScaled Feature Summary")

print(X_scaled.describe().round(2))

# ==============================================================================
# 5. INITIALIZE METRIC STORAGE
# ==============================================================================

section("CLUSTER OPTIMIZATION")

k_values = list(range(2, 11))

inertia_scores = []

silhouette_scores = []

davies_scores = []

calinski_scores = []

print("\nEvaluating Cluster Sizes...\n")

# ==============================================================================
# 6. EVALUATE CLUSTER SIZES
# ==============================================================================

print("\n" + "=" * 90)
print("K-MEANS CLUSTER EVALUATION")
print("=" * 90)

k_values = list(range(2, 11))

inertia_scores = []
silhouette_scores = []
davies_scores = []
calinski_scores = []

print("\nEvaluating cluster sizes...\n")

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
        max_iter=500
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    davies = davies_bouldin_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)

    inertia_scores.append(inertia)
    silhouette_scores.append(silhouette)
    davies_scores.append(davies)
    calinski_scores.append(calinski)

    print(
        f"k={k:<2} | "
        f"Inertia={inertia:10.2f} | "
        f"Silhouette={silhouette:.4f} | "
        f"Davies={davies:.4f} | "
        f"Calinski={calinski:.2f}"
    )

print("\n✓ Cluster evaluation completed.")

# ==============================================================================
# SELECT BEST K
# ==============================================================================

optimal_k = k_values[np.argmax(silhouette_scores)]

print("\n" + "=" * 90)
print(f"Optimal Number of Clusters = {optimal_k}")
print("=" * 90)

# ==============================================================================
# TRAIN FINAL K-MEANS MODEL
# ==============================================================================

print("\nTraining Final K-Means Model...\n")

kmeans_final = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=20,
    max_iter=500
)

cluster_labels = kmeans_final.fit_predict(X_scaled)

# Store cluster labels
rfm["Cluster"] = cluster_labels.astype(int)

# ==============================================================================
# Create Cluster Names
# ==============================================================================

cluster_name_map = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4"
}

rfm["Cluster_Name"] = "Cluster " + rfm["Cluster"].astype(str)

print("✓ Final clustering completed successfully.")

print("\nCluster Distribution")
print("-" * 40)
print(rfm["Cluster"].value_counts().sort_index())

print("\nRFM Columns")
print("-" * 40)
print(rfm.columns.tolist())

# Safety check
assert "Cluster" in rfm.columns, "Cluster column was not created."

# ==============================================================================
# SAVE CLUSTERED DATA
# ==============================================================================

rfm.to_csv(
    "data/processed/customer_segments.csv",
    index=False
)

print("\n✓ Customer segments saved successfully.")

# ==============================================================================
# 7. STORE RESULTS
# ==============================================================================

cluster_results = pd.DataFrame({

    "k": k_values,

    "Inertia": inertia_scores,

    "Silhouette": silhouette_scores,

    "Davies_Bouldin": davies_scores,

    "Calinski_Harabasz": calinski_scores

})

section("CLUSTER EVALUATION TABLE")

print(cluster_results.round(4))

print("\nHighest Silhouette Score")
print(rfm.columns.tolist())

best_row = cluster_results.loc[
    cluster_results["Silhouette"].idxmax()
]

print(best_row)

print("\nLowest Davies-Bouldin Score")

print(

    cluster_results.loc[
        cluster_results["Davies_Bouldin"].idxmin()
    ]

)

print("\nHighest Calinski-Harabasz Score")

print(

    cluster_results.loc[
        cluster_results["Calinski_Harabasz"].idxmax()
    ]

)


print("\n" + "=" * 90)
print("CLUSTER INTERPRETATION & BUSINESS RECOMMENDATIONS")
print("=" * 90)

overall_recency = rfm["Recency"].median()
overall_frequency = rfm["Frequency"].median()
overall_monetary = rfm["Monetary"].median()

cluster_recommendations = []

for cluster in sorted(rfm["Cluster"].unique()):

    cluster_df = rfm[rfm["Cluster"] == cluster]

    avg_r = cluster_df["Recency"].mean()
    avg_f = cluster_df["Frequency"].mean()
    avg_m = cluster_df["Monetary"].mean()

    revenue = cluster_df["Monetary"].sum()
    revenue_share = revenue / rfm["Monetary"].sum() * 100

    customers = len(cluster_df)

    # ----------------------------------------------------
    # Business-friendly cluster naming
    # ----------------------------------------------------

    if avg_r <= overall_recency and avg_f >= overall_frequency and avg_m >= overall_monetary:

        profile = "VIP Customers"

        recommendations = [
            "Offer premium loyalty benefits",
            "Provide early access to new products",
            "Launch exclusive membership campaigns",
            "Encourage referrals and advocacy"
        ]

    elif avg_r <= overall_recency and avg_f >= overall_frequency:

        profile = "Loyal Customers"

        recommendations = [
            "Upsell premium products",
            "Cross-sell complementary products",
            "Provide personalized recommendations",
            "Reward repeat purchases"
        ]

    elif avg_r <= overall_recency:

        profile = "Potential Customers"

        recommendations = [
            "Increase engagement through email campaigns",
            "Recommend frequently purchased products",
            "Provide limited-time offers",
            "Encourage second purchase"
        ]

    else:

        profile = "At-Risk Customers"

        recommendations = [
            "Launch win-back campaigns",
            "Offer personalized discounts",
            "Collect customer feedback",
            "Re-engage through email marketing"
        ]

    print("\n" + "-" * 90)
    print(f"Cluster {cluster} : {profile}")
    print("-" * 90)

    print(f"Customers            : {customers:,}")
    print(f"Revenue Contribution : {revenue_share:.2f}%")
    print(f"Average Recency      : {avg_r:.1f} days")
    print(f"Average Frequency    : {avg_f:.2f} orders")
    print(f"Average Monetary     : ${avg_m:,.2f}")

    print("\nRecommended Actions")

    for action in recommendations:
        print(f"  • {action}")

    cluster_recommendations.append({
        "Cluster": cluster,
        "Profile": profile,
        "Customers": customers,
        "Revenue_Share(%)": round(revenue_share,2),
        "Avg_Recency": round(avg_r,2),
        "Avg_Frequency": round(avg_f,2),
        "Avg_Monetary": round(avg_m,2)
    })

# =============================================================================
# Save Cluster Recommendation Report
# =============================================================================

cluster_report = pd.DataFrame(cluster_recommendations)

cluster_report.to_csv(
    "outputs/reports/cluster_business_recommendations.csv",
    index=False
)

print("\n✓ Cluster recommendation report saved.")


print("\n" + "=" * 90)
print("CUSTOMER SEGMENT VISUALIZATION")
print("=" * 90)

# -----------------------------------------------------------------------------
# PCA Projection
# -----------------------------------------------------------------------------

print("\nPerforming PCA dimensionality reduction...")

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(X_scaled)

rfm["PCA1"] = pca_components[:, 0]
rfm["PCA2"] = pca_components[:, 1]

explained_variance = pca.explained_variance_ratio_

print(f"PC1 Explained Variance : {explained_variance[0]*100:.2f}%")
print(f"PC2 Explained Variance : {explained_variance[1]*100:.2f}%")
print(f"Total Explained        : {explained_variance.sum()*100:.2f}%")

# -----------------------------------------------------------------------------
# Interactive PCA Scatter Plot
# -----------------------------------------------------------------------------

fig = px.scatter(
    rfm,
    x="PCA1",
    y="PCA2",
    color="Cluster_Name",
    hover_data=[
        "Customer_ID",
        "Recency",
        "Frequency",
        "Monetary"
    ],
    title="Customer Segments (PCA Projection)",
    labels={
        "PCA1": f"PC1 ({explained_variance[0]*100:.1f}% Variance)",
        "PCA2": f"PC2 ({explained_variance[1]*100:.1f}% Variance)"
    },
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig.update_traces(marker=dict(size=8))

fig.update_layout(
    template="plotly_white",
    title_x=0.5,
    height=650
)

fig.write_html("outputs/figures/19_customer_segments_pca.html")

print("✓ Saved : 19_customer_segments_pca.html")

# -----------------------------------------------------------------------------
# Static PCA Plot
# -----------------------------------------------------------------------------

plt.figure(figsize=(10,8))

sns.scatterplot(
    data=rfm,
    x="PCA1",
    y="PCA2",
    hue="Cluster_Name",
    palette="Set2",
    s=80,
    alpha=0.8
)

plt.title(
    "Customer Segments using PCA",
    fontsize=16,
    weight="bold"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(alpha=0.3)

plt.legend(title="Cluster")

plt.tight_layout()

plt.savefig(
    "outputs/figures/19_customer_segments_pca.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Saved : 19_customer_segments_pca.png")

# -----------------------------------------------------------------------------
# Cluster Size
# -----------------------------------------------------------------------------

cluster_size = (
    rfm["Cluster_Name"]
    .value_counts()
    .reset_index()
)

cluster_size.columns = ["Cluster","Customers"]

fig = px.bar(
    cluster_size,
    x="Cluster",
    y="Customers",
    color="Customers",
    text="Customers",
    title="Customer Count by Cluster",
    color_continuous_scale="Blues"
)

fig.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig.write_html(
    "outputs/figures/20_cluster_size.html"
)

print("✓ Saved : 20_cluster_size.html")

# -----------------------------------------------------------------------------
# Revenue Contribution
# -----------------------------------------------------------------------------

cluster_revenue = (
    rfm.groupby("Cluster_Name")["Monetary"]
       .sum()
       .sort_values(ascending=False)
       .reset_index()
)

cluster_revenue.columns = [
    "Cluster",
    "Revenue"
]

fig = px.bar(
    cluster_revenue,
    x="Cluster",
    y="Revenue",
    color="Revenue",
    text_auto=".2s",
    title="Revenue Contribution by Customer Cluster",
    color_continuous_scale="Viridis"
)

fig.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig.write_html(
    "outputs/figures/21_cluster_revenue.html"
)

print("✓ Saved : 21_cluster_revenue.html")

# -----------------------------------------------------------------------------
# Average RFM Metrics
# -----------------------------------------------------------------------------

cluster_profile = (
    rfm.groupby("Cluster_Name")[["Recency","Frequency","Monetary"]]
       .mean()
       .reset_index()
)

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Average Recency",
        "Average Frequency",
        "Average Monetary"
    ]
)

fig.add_trace(
    go.Bar(
        x=cluster_profile["Cluster_Name"],
        y=cluster_profile["Recency"],
        name="Recency"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Bar(
        x=cluster_profile["Cluster_Name"],
        y=cluster_profile["Frequency"],
        name="Frequency"
    ),
    row=1,
    col=2
)

fig.add_trace(
    go.Bar(
        x=cluster_profile["Cluster_Name"],
        y=cluster_profile["Monetary"],
        name="Monetary"
    ),
    row=1,
    col=3
)

fig.update_layout(
    template="plotly_white",
    height=500,
    showlegend=False,
    title="Average RFM Metrics by Cluster",
    title_x=0.5
)

fig.write_html(
    "outputs/figures/22_cluster_rfm_profile.html"
)

print("✓ Saved : 22_cluster_rfm_profile.html")

# =============================================================================
# PART D : MARKET BASKET ANALYSIS (ASSOCIATION RULE MINING)
# =============================================================================

print("\n" + "=" * 90)
print("MARKET BASKET ANALYSIS")
print("=" * 90)

if MBA_AVAILABLE and {"Order_ID", "Product_ID"}.issubset(df.columns):

    try:

        # ---------------------------------------------------------------------
        # Prepare Transaction Data
        # ---------------------------------------------------------------------

        print("\nPreparing transaction data...")

        transactions = (
            df.groupby("Order_ID")["Product_ID"]
              .apply(lambda x: list(set(x.astype(str))))
              .tolist()
        )

        print(f"Total Transactions : {len(transactions):,}")

        # ---------------------------------------------------------------------
        # Transaction Encoding
        # ---------------------------------------------------------------------

        encoder = TransactionEncoder()

        encoded = encoder.fit(transactions).transform(transactions)

        basket = pd.DataFrame(
            encoded,
            columns=encoder.columns_
        )

        print(f"Unique Products : {basket.shape[1]}")

        # ---------------------------------------------------------------------
        # Frequent Itemsets
        # ---------------------------------------------------------------------

        print("\nGenerating frequent itemsets...")

        frequent_itemsets = apriori(
            basket,
            min_support=0.01,
            use_colnames=True
        )

        frequent_itemsets["Item_Count"] = frequent_itemsets["itemsets"].apply(len)

        frequent_itemsets = frequent_itemsets.sort_values(
            "support",
            ascending=False
        )

        print(f"Frequent Itemsets Found : {len(frequent_itemsets):,}")

        # ---------------------------------------------------------------------
        # Association Rules
        # ---------------------------------------------------------------------

        print("Generating association rules...")

        rules = association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=1.2
        )

        rules = rules.sort_values(
            by=["lift", "confidence"],
            ascending=False
        )

        # Convert frozensets into readable strings

        rules["Antecedent"] = rules["antecedents"].apply(
            lambda x: ", ".join(list(x))
        )

        rules["Consequent"] = rules["consequents"].apply(
            lambda x: ", ".join(list(x))
        )

        rules_display = rules[
            [
                "Antecedent",
                "Consequent",
                "support",
                "confidence",
                "lift"
            ]
        ].copy()

        print(f"\nAssociation Rules Generated : {len(rules_display):,}")

        print("\nTop 15 Association Rules\n")

        print(
            rules_display.head(15).round(3)
        )

        # ---------------------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------------------

        fig = px.scatter(
            rules_display.head(100),
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_name="Antecedent",
            hover_data=["Consequent"],
            title="Association Rules (Support vs Confidence)",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            height=650
        )

        fig.write_html(
            "outputs/figures/23_association_rules.html"
        )

        print("✓ Saved : 23_association_rules.html")

        # ---------------------------------------------------------------------
        # Top Frequent Itemsets
        # ---------------------------------------------------------------------

        top_itemsets = (
            frequent_itemsets
            .query("Item_Count == 1")
            .head(15)
            .copy()
        )

        top_itemsets["Product"] = top_itemsets["itemsets"].apply(
            lambda x: list(x)[0]
        )

        fig = px.bar(
            top_itemsets,
            x="Product",
            y="support",
            color="support",
            text="support",
            title="Top Selling Individual Products (Support)",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5
        )

        fig.write_html(
            "outputs/figures/24_top_products_support.html"
        )

        print("✓ Saved : 24_top_products_support.html")

        # ---------------------------------------------------------------------
        # Save Reports
        # ---------------------------------------------------------------------

        frequent_itemsets.to_csv(
            "outputs/reports/frequent_itemsets.csv",
            index=False
        )

        rules_display.to_csv(
            "outputs/reports/association_rules.csv",
            index=False
        )

        print("\n✓ Frequent itemsets saved")
        print("✓ Association rules saved")

        # ---------------------------------------------------------------------
        # Business Insights
        # ---------------------------------------------------------------------

        print("\nBusiness Insights")

        if len(rules_display) > 0:

            best_rule = rules_display.iloc[0]

            print(
                f"• Customers buying [{best_rule['Antecedent']}] "
                f"also tend to buy [{best_rule['Consequent']}]"
            )

            print(
                f"• Confidence : {best_rule['confidence']:.2%}"
            )

            print(
                f"• Lift : {best_rule['lift']:.2f}"
            )

            print(
                "• Recommendation : Bundle these products or "
                "recommend them together."
            )

        else:

            print("No strong association rules were discovered.")

    except Exception as e:

        print(f"\nMarket Basket Analysis skipped : {e}")

else:

    print("\nMarket Basket Analysis skipped.")
    print("Reason:")
    print(" • mlxtend library is not installed")
    print(" • OR Product_ID / Order_ID columns are unavailable")

# =============================================================================
# PART E : COHORT ANALYSIS (CUSTOMER RETENTION)
# =============================================================================

print("\n" + "=" * 90)
print("COHORT ANALYSIS")
print("=" * 90)

# -----------------------------------------------------------------------------
# Create Cohort Dataset
# -----------------------------------------------------------------------------

cohort_df = df.copy()

cohort_df["Order_Month"] = cohort_df["Order_Date"].dt.to_period("M")

cohort_df["Cohort_Month"] = (
    cohort_df.groupby("Customer_ID")["Order_Date"]
             .transform("min")
             .dt.to_period("M")
)

# -----------------------------------------------------------------------------
# Cohort Index (Months Since First Purchase)
# -----------------------------------------------------------------------------

cohort_df["Cohort_Index"] = (
    (cohort_df["Order_Month"] - cohort_df["Cohort_Month"])
    .apply(lambda x: x.n)
)

# -----------------------------------------------------------------------------
# Cohort Size
# -----------------------------------------------------------------------------

cohort_size = (
    cohort_df.groupby("Cohort_Month")["Customer_ID"]
             .nunique()
)

print(f"\nTotal Cohorts : {len(cohort_size)}")

print("\nRecent Cohort Sizes")

print(cohort_size.tail())

# -----------------------------------------------------------------------------
# Customer Retention Matrix
# -----------------------------------------------------------------------------

retention = (
    cohort_df.groupby(
        ["Cohort_Month", "Cohort_Index"]
    )["Customer_ID"]
    .nunique()
    .reset_index()
)

retention_matrix = retention.pivot(
    index="Cohort_Month",
    columns="Cohort_Index",
    values="Customer_ID"
)

retention_rate = (
    retention_matrix.divide(cohort_size, axis=0) * 100
)

print("\nCustomer Retention Matrix (%)")

print(retention_rate.round(1))

# -----------------------------------------------------------------------------
# Retention Heatmap
# -----------------------------------------------------------------------------

plt.figure(figsize=(15, 8))

sns.heatmap(
    retention_rate,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Retention (%)"},
    vmin=0,
    vmax=100
)

plt.title(
    "Customer Cohort Retention Analysis",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Months Since First Purchase",
    fontsize=12
)

plt.ylabel(
    "Customer Cohort",
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/25_cohort_retention_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Saved : 25_cohort_retention_heatmap.png")

# -----------------------------------------------------------------------------
# Average Retention Curve
# -----------------------------------------------------------------------------

avg_retention = retention_rate.mean(axis=0)

plt.figure(figsize=(10, 6))

plt.plot(
    avg_retention.index,
    avg_retention.values,
    marker="o",
    linewidth=2
)

plt.title(
    "Average Customer Retention Curve",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Months After First Purchase")

plt.ylabel("Retention Rate (%)")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/figures/26_average_retention_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Saved : 26_average_retention_curve.png")

# -----------------------------------------------------------------------------
# Monthly Cohort Revenue
# -----------------------------------------------------------------------------

cohort_revenue = (
    cohort_df.groupby("Cohort_Month")["Sales"]
             .sum()
             .reset_index()
)

cohort_revenue["Cohort_Month"] = (
    cohort_revenue["Cohort_Month"].astype(str)
)

fig = px.bar(
    cohort_revenue,
    x="Cohort_Month",
    y="Sales",
    color="Sales",
    title="Revenue Generated by Customer Cohorts",
    color_continuous_scale="Blues"
)

fig.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig.write_html(
    "outputs/figures/27_cohort_revenue.html"
)

print("✓ Saved : 27_cohort_revenue.html")

# -----------------------------------------------------------------------------
# Business Insights
# -----------------------------------------------------------------------------

print("\nBusiness Insights")
print("-" * 60)

avg_first_month = retention_rate.iloc[:, 0].mean()

print(f"Average Month-0 Retention : {avg_first_month:.2f}%")

if retention_rate.shape[1] > 1:

    avg_second_month = retention_rate.iloc[:, 1].mean()

    print(f"Average Month-1 Retention : {avg_second_month:.2f}%")

if retention_rate.shape[1] > 2:

    avg_third_month = retention_rate.iloc[:, 2].mean()

    print(f"Average Month-2 Retention : {avg_third_month:.2f}%")

print("\nRecommendations")

print("• Improve customer onboarding during the first purchase.")

print("• Launch retention campaigns after the first month.")

print("• Reward repeat purchases with loyalty benefits.")

print("• Target cohorts showing rapid decline.")

# -----------------------------------------------------------------------------
# Save Reports
# -----------------------------------------------------------------------------

retention_rate.to_csv(
    "outputs/reports/cohort_retention_matrix.csv"
)

cohort_size.to_csv(
    "outputs/reports/cohort_sizes.csv"
)

print("\n✓ Cohort retention matrix saved.")

print("✓ Cohort size report saved.")

# =============================================================================
# PART D : CUSTOMER LIFETIME VALUE (CLV)
# =============================================================================

print("\n" + "="*90)
print("CUSTOMER LIFETIME VALUE (CLV)")
print("="*90)

# ------------------------------------------------------------------
# Calculate Average Order Value
# ------------------------------------------------------------------

customer_orders = df.groupby("Customer_ID").agg({
    "Sales": "sum",
    "Order_ID": "nunique"
}).reset_index()

customer_orders.rename(columns={
    "Sales": "Total_Sales",
    "Order_ID": "Total_Orders"
}, inplace=True)

customer_orders["Average_Order_Value"] = (
    customer_orders["Total_Sales"] /
    customer_orders["Total_Orders"]
)

# ------------------------------------------------------------------
# Purchase Frequency
# ------------------------------------------------------------------

total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()

purchase_frequency = total_orders / total_customers

customer_orders["Purchase_Frequency"] = purchase_frequency

# ------------------------------------------------------------------
# Profit Margin Assumption
# ------------------------------------------------------------------

profit_margin = 0.25

customer_orders["Profit_Margin"] = profit_margin

# ------------------------------------------------------------------
# Simple CLV Formula
# ------------------------------------------------------------------

customer_orders["CLV_Simple"] = (
    customer_orders["Average_Order_Value"] *
    customer_orders["Purchase_Frequency"] *
    customer_orders["Profit_Margin"]
)

# ------------------------------------------------------------------
# Merge with RFM
# ------------------------------------------------------------------

customer_clv = pd.merge(
    rfm,
    customer_orders[
        [
            "Customer_ID",
            "Average_Order_Value",
            "Purchase_Frequency",
            "Profit_Margin",
            "CLV_Simple"
        ]
    ],
    on="Customer_ID",
    how="left"
)

# ------------------------------------------------------------------
# Fill Missing Values
# ------------------------------------------------------------------

customer_clv.fillna(0, inplace=True)

# ------------------------------------------------------------------
# Display Summary
# ------------------------------------------------------------------

print("\nCustomer CLV Summary")

print(customer_clv[
    [
        "Customer_ID",
        "Average_Order_Value",
        "Purchase_Frequency",
        "CLV_Simple"
    ]
].head())

print("\nAverage CLV : ${:,.2f}".format(
    customer_clv["CLV_Simple"].mean()
))

print("Maximum CLV : ${:,.2f}".format(
    customer_clv["CLV_Simple"].max()
))

print("Minimum CLV : ${:,.2f}".format(
    customer_clv["CLV_Simple"].min()
))

# ------------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------------

customer_clv.to_csv(
    "data/processed/customer_clv.csv",
    index=False
)

print("\n✓ customer_clv.csv saved successfully.")
print("Location : data/processed/customer_clv.csv")