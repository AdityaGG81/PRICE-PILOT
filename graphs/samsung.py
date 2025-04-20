import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import pymysql
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="Samsung S23 Ultra Price Tracker")

# Title & styling
st.markdown("""
    <style>
        .title {
            font-weight: bold;
            font-size: 32px;
        }
        .update {
            font-size: 16px;
            text-align: center;
            margin-top: -20px;
        }
        .stSelectbox label {
            font-weight: bold;
        }
    </style>
    <center><div class='title'>SAMSUNG GALAXY S23 ULTRA 256GB PRICE COMPARISON</div></center>
""", unsafe_allow_html=True)

# Last updated
today_str = datetime.datetime.now().strftime("%d %B %Y")
st.markdown(f"<p class='update'><b>Last updated on:</b> {today_str}</p>", unsafe_allow_html=True)

# Filter selection
filter_col1, filter_col2, filter_col3 = st.columns([0.7, 0.1, 0.2])
with filter_col3:
    time_filter = st.selectbox(
        "Select Time Period",
        ["All Time", "Last Week", "Last Month", "Last Year"],
        index=0
    )

# MySQL DB Connection
try:
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="scraper_db"
    )
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Load data
try:
    df = pd.read_sql("SELECT * FROM view_samsung_galaxy_s23_ultra_256gb", connection)
    connection.close()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if df.empty:
    st.warning("No data available for Samsung S23 Ultra.")
    st.stop()

# Convert date column
df['date'] = pd.to_datetime(df['date'])
today = datetime.datetime.now().date()

# Apply time filter
if time_filter == "Last Week":
    df = df[df['date'] >= pd.Timestamp(today - timedelta(days=7))]
elif time_filter == "Last Month":
    df = df[df['date'] >= pd.Timestamp(today - timedelta(days=30))]
elif time_filter == "Last Year":
    df = df[df['date'] >= pd.Timestamp(today - timedelta(days=365))]

if df.empty:
    st.warning(f"No data for selected time period: {time_filter}")
    st.stop()

# Color settings
custom_colors = {
    "Amazon": "#064adc",
    "Flipkart": "#fff033",
    "Amazon (Predicted)": "#00FFFF",
    "Flipkart (Predicted)": "#32CD32"
}

# Predict prices
def predict_next_days(df, retailer, days=4):
    data = df[df["source"] == retailer].copy()
    data["date_ordinal"] = data["date"].map(datetime.datetime.toordinal)
    
    X = data[["date_ordinal"]]
    y = data["current_price"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = data["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predicted_prices = model.predict(future_ordinals)
    
    predicted_df = pd.DataFrame({
        "date": future_dates,
        "current_price": predicted_prices,
        "source": f"{retailer} (Predicted)"
    })
    return predicted_df

amazon_pred = predict_next_days(df, "Amazon")
flipkart_pred = predict_next_days(df, "Flipkart")
predicted_df = pd.concat([amazon_pred, flipkart_pred])

# Merge predicted + actual
combined_df = pd.concat([df, predicted_df])

# Bar Chart
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:
    st.markdown("<h3 style='text-align: center;'>Price Comparison: Amazon vs Flipkart</h3>", unsafe_allow_html=True)
    bar_fig = px.bar(
        df,
        x="date",
        y="current_price",
        color="source",
        barmode="group",
        labels={"current_price": "Current Price (₹)", "date": "Date", "source": "Retailer"},
        color_discrete_map=custom_colors,
        template="plotly_dark",
        height=500
    )
    bar_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# Price Stats + Latest Prices
col_stats, col_latest = st.columns(2)

with col_stats:
    st.subheader("Price Statistics")
    stats = df.groupby("source")["current_price"].agg(['min', 'max', 'mean']).reset_index()
    stats['mean'] = stats['mean'].round(2)
    stats.columns = ['Source', 'Min Price (₹)', 'Max Price (₹)', 'Avg Price (₹)']
    st.dataframe(stats, hide_index=True, use_container_width=True)

with col_latest:
    st.subheader("Latest Prices")
    max_date = df["date"].max()
    latest_prices = df[df["date"] == max_date][["source", "current_price"]]
    latest_prices.columns = ['Source', 'Current Price (₹)']
    st.dataframe(latest_prices, hide_index=True, use_container_width=True)

# Line Chart with Predicted Data
st.subheader("Price Trends (with 4-Day Predictions)")
line_fig = px.line(
    combined_df,
    x="date",
    y="current_price",
    color="source",
    labels={"current_price": "Price (₹)", "date": "Date", "source": "Retailer"},
    color_discrete_map=custom_colors
)
line_fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(line_fig, use_container_width=True)

# Prediction Table
st.subheader("Predicted Prices for Next 4 Days")
prediction_table = predicted_df.copy()
prediction_table['date'] = prediction_table['date'].dt.strftime("%d-%b-%Y")
prediction_table['current_price'] = prediction_table['current_price'].round(2)
prediction_table.columns = ['Date', 'Predicted Price (₹)', 'Source']
st.dataframe(prediction_table, hide_index=True, use_container_width=True)
