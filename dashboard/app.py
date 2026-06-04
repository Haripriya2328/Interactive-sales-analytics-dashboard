import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Sales Analytics Dashboard",layout="wide")


# read dataset
data = pd.read_csv("data/sales_data_medium.csv")

# convert date column
data["date"] = pd.to_datetime(data["date"])
data["day_number"]=np.arange(len(data))
data["month"]=data["date"].dt.month_name()
# ================= MACHINE LEARNING =================

# input feature
X = data[["day_number"]]

# target variable
y = data["amount"]

# create model
model = LinearRegression()

# train model
model.fit(X, y)

# future prediction
future_day = [[len(data)]]

predicted_sales = model.predict(future_day)
all_predictions=model.predict(X)
# dashboard title
st.title("📊 Interactive Sales Analytics Dashboard")

st.markdown("""This dashboard analyzes product, region, and time-based sales trends and predicts future sales using Machine Learning.""")
st.sidebar.title("📌 Dashboard Filters")
selected_product=st.sidebar.selectbox("Select Product",["All"]+list(data["product"].unique()))
selected_region=st.sidebar.selectbox("Select Region",["All"]+list(data["region"].unique()))
st.sidebar.success("Filters Applied Successfully")

filtered_data=data.copy()
if selected_product!="All":
    filtered_data=filtered_data[filtered_data["product"]==selected_product]

if selected_region != "All":
    filtered_data=filtered_data[filtered_data["region"]==selected_region]

total_sales=filtered_data["amount"].sum()
best_product=(filtered_data.groupby("product")["amount"].sum().idxmax())
best_region=(filtered_data.groupby("region")["amount"].sum().idxmax())

#====KPI Cards======
st.subheader("📌 Key Performance Indicators")
col1,col2,col3=st.columns(3)
col1.metric("Total Sales",f"{total_sales:,}")
col2.metric("Best Product",best_product)
col3.metric("Best Region",best_region)

st.subheader("🤖 Predicted Next Day Sales")
st.markdown("----")
st.subheader("📈 Actual vs Predicted Sales Trend")


st.success(
    f"Predicted Sales: ₹ {predicted_sales[0]:,.2f}")
# analysis
product_sales = filtered_data.groupby("product")["amount"].sum()
region_sales = filtered_data.groupby("region")["amount"].sum()
date_sales = filtered_data.groupby("date")["amount"].sum()
monthly_sales=(filtered_data.groupby("month")["amount"].sum())

# create figure
fig = plt.figure(figsize=(12, 8))

# ---------- Date-wise Sales ----------
plt.subplot(2, 1, 1)

plt.plot(
    date_sales.index,
    date_sales.values,
    color="orange"
)

plt.title("Date-wise Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.grid(True)
plt.xticks(rotation=45)

plt.gca().yaxis.set_major_formatter(
    StrMethodFormatter('{x:,.0f}')
)

# ---------- Product-wise Sales ----------
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
    StrMethodFormatter('{x:,.0f}')
)

for bar in bars1:

    y = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f'{int(y):,}',
        ha='center',
        va='bottom'
    )

# ---------- Region-wise Sales ----------
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
    StrMethodFormatter('{x:,.0f}')
)

for bar in bars2:

    y = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f'{int(y):,}',
        ha='center',
        va='bottom'
    )
#------------Monthly wise sales-------------
st.subheader("Monthly Sales Analytics")
fig2=plt.figure(figsize=(8,4))
plt.bar(monthly_sales.index,monthly_sales.values,color="skyblue")
plt.title("Month-wise Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
st.pyplot(fig2)

st.subheader("🥧 Product Sales Distribution")
fig3 = plt.figure(figsize=(6, 6))
plt.pie(
    product_sales.values,
    labels=product_sales.index,
    autopct='%1.1f%%'
)
plt.title("Product-wise sales Distribution")
st.pyplot(fig3)

fig4=plt.figure(figsize=(10,5))
plt.plot(data["day_number"],label="Actual Sales",color="blue")
plt.plot(data["day_number"],all_predictions,label="Predicted Trend",color="red")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Day Number")
plt.ylabel("Sales")

plt.legend()

plt.grid(True)
st.pyplot(fig4)

plt.tight_layout()

# show dashboard
st.markdown("----")
st.pyplot(fig)
st.subheader("📋 Filtered Sales Data")
st.dataframe(filtered_data)
csv=filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Filtered Data",data=csv,file_name="filtered_sales_data.csv",mime="text/csv")
st.success("Dashboard successfully built using Python, Pandas, Matplotlib, Streamlit, and Machine Learning.")