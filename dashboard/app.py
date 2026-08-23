import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.linear_model import LinearRegression
import numpy as np


# ================= PAGE CONFIGURATION =================

st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide"
)


# ================= READ DATASET =================

data = pd.read_csv("data/sales_data_medium.csv")


# ================= DATA PREPROCESSING =================

# Convert date column to datetime
data["date"] = pd.to_datetime(data["date"])

# Create day number
data["day_number"] = np.arange(len(data))

# Create month column
data["month"] = data["date"].dt.month_name()


# ================= MACHINE LEARNING =================

# Input feature
X = data[["day_number"]]

# Target variable
y = data["amount"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict next day's sales
future_day = [[len(data)]]

predicted_sales = model.predict(future_day)

# Predict sales for existing days
all_predictions = model.predict(X)


# ================= DASHBOARD TITLE =================

st.title("📊 Interactive Sales Analytics Dashboard")

st.markdown(
    "This dashboard analyzes product, region, and time-based sales "
    "trends and predicts future sales using Machine Learning."
)


# ================= SIDEBAR FILTERS =================

st.sidebar.title("📌 Dashboard Filters")

selected_product = st.sidebar.selectbox(
    "Select Product",
    ["All"] + list(data["product"].unique())
)

selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + list(data["region"].unique())
)

st.sidebar.success("Filters Applied Successfully")


# ================= FILTER DATA =================

filtered_data = data.copy()

if selected_product != "All":
    filtered_data = filtered_data[
        filtered_data["product"] == selected_product
    ]

if selected_region != "All":
    filtered_data = filtered_data[
        filtered_data["region"] == selected_region
    ]


# ================= KPI CALCULATIONS =================

total_sales = filtered_data["amount"].sum()

best_product = (
    filtered_data.groupby("product")["amount"]
    .sum()
    .idxmax()
)

best_region = (
    filtered_data.groupby("region")["amount"]
    .sum()
    .idxmax()
)


# ================= KPI CARDS =================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sales",
    f"{total_sales:,.0f}"
)

col2.metric(
    "Best Product",
    best_product
)

col3.metric(
    "Best Region",
    best_region
)


# ================= PREDICTION =================

st.subheader("🤖 Predicted Next Day Sales")

st.markdown("----")

st.subheader("📈 Actual vs Predicted Sales Trend")

st.success(
    f"Predicted Sales: ₹ {predicted_sales[0]:,.2f}"
)


# ================= SALES ANALYSIS =================

product_sales = (
    filtered_data.groupby("product")["amount"]
    .sum()
)

region_sales = (
    filtered_data.groupby("region")["amount"]
    .sum()
)

date_sales = (
    filtered_data.groupby("date")["amount"]
    .sum()
    .sort_index()
)

monthly_sales = (
    filtered_data.groupby("month")["amount"]
    .sum()
)


# =========================================================
# MAIN SALES CHART
# =========================================================

fig = plt.figure(figsize=(12, 10))


# ================= DATE-WISE SALES =================

plt.subplot(2, 2, (1, 2))

plt.plot(
    date_sales.index,
    date_sales.values,
    marker="o",
    color="orange"
)

plt.title("Date-wise Sales Trend")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.grid(True)

# Rotate date labels
plt.xticks(
    rotation=45,
    ha="right"
)

# Format y-axis
plt.gca().yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)


# ================= PRODUCT-WISE SALES =================

plt.subplot(2, 2, 3)

bars1 = plt.bar(
    product_sales.index,
    product_sales.values,
    color=["olive", "teal", "brown"]
)

plt.title("Product-wise Sales")

plt.xlabel("Product")

plt.ylabel("Sales")

plt.gca().yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)


# Add values above bars
for bar in bars1:

    y = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f"{int(y):,}",
        ha="center",
        va="bottom"
    )


# ================= REGION-WISE SALES =================

plt.subplot(2, 2, 4)

bars2 = plt.bar(
    region_sales.index,
    region_sales.values,
    color=["cyan", "magenta", "orange", "purple"]
)

plt.title("Region-wise Sales")

plt.xlabel("Region")

plt.ylabel("Sales")

plt.gca().yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)


# Add values above bars
for bar in bars2:

    y = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f"{int(y):,}",
        ha="center",
        va="bottom"
    )


# Adjust main chart spacing
fig.tight_layout(pad=3.0)

# Add vertical spacing between rows
fig.subplots_adjust(
    hspace=0.6
)


# ================= MONTHLY SALES =================

st.subheader("📅 Monthly Sales Analytics")

fig2 = plt.figure(figsize=(8, 4))

plt.bar(
    monthly_sales.index,
    monthly_sales.values,
    color="skyblue"
)

plt.title("Month-wise Sales")

plt.xlabel("Month")

plt.ylabel("Sales")

plt.gca().yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig2)


# ================= PRODUCT DISTRIBUTION =================

st.subheader("🥧 Product Sales Distribution")

fig3 = plt.figure(figsize=(6, 6))

plt.pie(
    product_sales.values,
    labels=product_sales.index,
    autopct="%1.1f%%"
)

plt.title("Product-wise Sales Distribution")

plt.tight_layout()

st.pyplot(fig3)


# ================= ACTUAL VS PREDICTED =================

fig4 = plt.figure(figsize=(10, 5))

plt.plot(
    data["day_number"],
    data["amount"],
    label="Actual Sales",
    color="blue"
)

plt.plot(
    data["day_number"],
    all_predictions,
    label="Predicted Trend",
    color="red"
)

plt.title("Actual vs Predicted Sales")

plt.xlabel("Day Number")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.tight_layout()

st.pyplot(fig4)


# ================= SHOW MAIN DASHBOARD CHART =================

st.markdown("----")

st.pyplot(fig)


# ================= FILTERED DATA =================

st.subheader("📋 Filtered Sales Data")

st.dataframe(
    filtered_data
)


# ================= DOWNLOAD BUTTON =================

csv = filtered_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)


# ================= SUCCESS MESSAGE =================

st.success(
    "Dashboard successfully built using Python, Pandas, "
    "Matplotlib, Streamlit, and Machine Learning."
)