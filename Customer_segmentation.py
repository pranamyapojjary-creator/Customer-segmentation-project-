# ============================================================
# CUSTOMER SEGMENTATION PROJECT
# Using K-Means Clustering and Machine Learning
# ============================================================

# 1. Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

# Set visualization style
sns.set_theme(style="whitegrid")

print("=" * 60)
print("       CUSTOMER SEGMENTATION PROJECT")
print("=" * 60)


# 2. Create a sample customer dataset
# You can replace this dataset with real customer data later.

np.random.seed(42)

number_of_customers = 300

customer_data = pd.DataFrame({
    "CustomerID": range(1, number_of_customers + 1),

    "Age": np.random.randint(18, 70, number_of_customers),

    "AnnualIncome": np.random.randint(
        15000, 120000, number_of_customers
    ),

    "SpendingScore": np.random.randint(
        1, 101, number_of_customers
    ),

    "PurchasesPerMonth": np.random.randint(
        1, 21, number_of_customers
    ),

    "AverageOrderValue": np.random.randint(
        200, 10001, number_of_customers
    )
})

print("\nFirst five customers:")
print(customer_data.head())

print("\nDataset information:")
print(customer_data.info())

print("\nMissing values:")
print(customer_data.isnull().sum())


# 3. Exploratory Data Analysis

print("\nCustomer Dataset Statistics:")
print(customer_data.describe())

# Age distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    customer_data["Age"],
    bins=15,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# Annual income distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    customer_data["AnnualIncome"],
    bins=20,
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# 4. Select features for customer segmentation

features = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchasesPerMonth",
    "AverageOrderValue"
]

X = customer_data[features]


# 5. Standardize the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nCustomer data standardized successfully.")


# 6. Find the optimal number of clusters using the Elbow Method

inertia_values = []

K = range(2, 11)

for k in K:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia_values.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    list(K),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(list(K))
plt.tight_layout()
plt.show()


# 7. Apply K-Means Clustering

number_of_clusters = 4

kmeans = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed.")


# 8. Evaluate the clustering model

silhouette_avg = silhouette_score(
    X_scaled,
    customer_data["Cluster"]
)

print(
    "\nSilhouette Score:",
    round(silhouette_avg, 3)
)

print(
    "Note: A higher silhouette score generally indicates "
    "better-separated clusters."
)


# 9. Name customer segments based on their characteristics

cluster_summary = customer_data.groupby("Cluster").agg({
    "Age": "mean",
    "AnnualIncome": "mean",
    "SpendingScore": "mean",
    "PurchasesPerMonth": "mean",
    "AverageOrderValue": "mean",
    "CustomerID": "count"
}).rename(
    columns={"CustomerID": "NumberOfCustomers"}
)

# Identify clusters with relatively high or low income
# and spending using the generated data.

income_median = cluster_summary["AnnualIncome"].median()
spending_median = cluster_summary["SpendingScore"].median()

def assign_segment(row):

    high_income = row["AnnualIncome"] >= income_median
    high_spending = row["SpendingScore"] >= spending_median

    if high_income and high_spending:
        return "Premium Customers"

    elif high_income and not high_spending:
        return "Careful Spenders"

    elif not high_income and high_spending:
        return "Active Spenders"

    else:
        return "Budget Customers"


cluster_summary["CustomerSegment"] = cluster_summary.apply(
    assign_segment,
    axis=1
)

segment_mapping = cluster_summary["CustomerSegment"].to_dict()

customer_data["CustomerSegment"] = customer_data["Cluster"].map(
    segment_mapping
)

print("\nCUSTOMER SEGMENT SUMMARY")
print("=" * 60)

print(
    cluster_summary.round(2).to_string()
)


# 10. Visualize customer segments

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=customer_data,
    x="AnnualIncome",
    y="SpendingScore",
    hue="CustomerSegment",
    palette="Set2",
    s=80
)

plt.title("Customer Segmentation by Income and Spending")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(
    title="Customer Segment",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()


# 11. Visualize the number of customers in each segment

plt.figure(figsize=(9, 5))

sns.countplot(
    data=customer_data,
    x="CustomerSegment",
    hue="CustomerSegment",
    order=customer_data["CustomerSegment"].value_counts().index,
    palette="Set2",
    legend=False
)

plt.title("Number of Customers in Each Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# 12. Compare purchasing behavior across segments

segment_behavior = customer_data.groupby(
    "CustomerSegment"
)[
    [
        "PurchasesPerMonth",
        "AverageOrderValue",
        "SpendingScore"
    ]
].mean()

segment_behavior.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Purchasing Behavior by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Value")
plt.xticks(rotation=15)
plt.legend(title="Metric")

plt.tight_layout()
plt.show()


# 13. Display customer insights

print("\n" + "=" * 60)
print("             CUSTOMER INSIGHTS")
print("=" * 60)

for segment, row in cluster_summary.iterrows():

    print(f"\nSegment: {row['CustomerSegment']}")

    print(
        "Number of customers:",
        int(row["NumberOfCustomers"])
    )

    print(
        "Average annual income: ₹",
        round(row["AnnualIncome"], 2)
    )

    print(
        "Average spending score:",
        round(row["SpendingScore"], 2)
    )

    print(
        "Average monthly purchases:",
        round(row["PurchasesPerMonth"], 2)
    )

    print(
        "Average order value: ₹",
        round(row["AverageOrderValue"], 2)
    )


# 14. Export the segmented customer dataset

customer_data.to_csv(
    "customer_segmentation_results.csv",
    index=False
)

cluster_summary.to_csv(
    "customer_segment_summary.csv"
)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles generated:")
print("1. customer_segmentation_results.csv")
print("2. customer_segment_summary.csv")
