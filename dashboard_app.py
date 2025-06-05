import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

# Streamlit page config
st.set_page_config(page_title="📊 Sales Dashboard & Forecasting", layout="wide")
st.title("📈 Sales Dashboard & Forecasting")

# --- Load and cache data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Your Data")
regions = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
categories = st.sidebar.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())
date_range = st.sidebar.date_input("Select Date Range", [df['Order Date'].min(), df['Order Date'].max()])

# Filter data
filtered_df = df[
    (df['Region'].isin(regions)) &
    (df['Category'].isin(categories)) &
    (df['Order Date'] >= pd.to_datetime(date_range[0])) &
    (df['Order Date'] <= pd.to_datetime(date_range[1]))
]

# --- KPIs ---
st.markdown("## 📌 Key Performance Indicators")
total_sales = filtered_df['Sales'].sum()
total_orders = filtered_df.shape[0]
avg_order_value = total_sales / total_orders if total_orders > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📦 Total Orders", total_orders)
col3.metric("💳 Avg Order Value", f"${avg_order_value:,.2f}")

# --- Sales Trend ---
st.markdown("## 📊 Sales Over Time")
daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

fig1, ax1 = plt.subplots()
ax1.plot(daily_sales['ds'], daily_sales['y'], color='blue')
ax1.set_title("Sales Trend")
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
st.pyplot(fig1)

# --- Forecasting ---
st.markdown("## 🔮 Forecast Future Sales")
forecast_days = st.slider("Select Forecast Days", 30, 180, 90)

model = Prophet()
model.fit(daily_sales)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

fig2 = model.plot(forecast)
st.pyplot(fig2)

if st.checkbox("Show Forecast Components"):
    st.pyplot(model.plot_components(forecast))

# --- Additional Charts ---
st.markdown("## 📦 Sales by Category (Bar Chart)")
category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
fig3 = px.bar(category_sales, x='Category', y='Sales', color='Category', title="Sales by Category")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("## 🌍 Sales by Region (Pie Chart)")
region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
fig4 = px.pie(region_sales, names='Region', values='Sales', title='Sales Distribution by Region')
st.plotly_chart(fig4, use_container_width=True)

# --- Download filtered data ---
st.markdown("## ⬇️ Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "filtered_sales.csv", "text/csv")
