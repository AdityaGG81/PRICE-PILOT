import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import pymysql
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# Connect to MySQL
try:
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="scraper_db"
    )
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# Page config
st.set_page_config(layout="wide", page_title="ASUS TUF A15 Price Tracker")

# Title and styling
html_title = """
    <style>
    .title-test {
        font-weight: bold;
        padding: 5px;
        border-radius: 6px;
        margin-top: -50px;
    }
    .update-text {
        font-size: 16px;
        text-align: center;
        margin-top: -30px;
    }
    </style>
    <center><h1 class="title-test">ASUS TUF A15 GAMING LAPTOP PRICE COMPARISON</h1></center>
"""
st.markdown(html_title, unsafe_allow_html=True)

# Last updated date
box_date = datetime.datetime.now().strftime("%d %B %Y")
st.markdown(f"<p class='update-text'><b>Last updated on:</b> {box_date}</p>", unsafe_allow_html=True)

# Time filter
_, _, col_filter = st.columns([0.7, 0.1, 0.2])
with col_filter:
    time_filter = st.selectbox("Time Period", ["All Time", "Last Week", "Last Month", "Last Year"])

# Read data
try:
    query = "SELECT * FROM view_asus_tuf_a15_gaming_laptop"
    df = pd.read_sql(query, connection)
    connection.close()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    if 'connection' in locals() and connection.open:
        connection.close()
    st.stop()

if df.empty:
    st.warning("No data available for ASUS TUF A15.")
    st.stop()

# Preprocess
df['date'] = pd.to_datetime(df['date'])
today = datetime.datetime.now().date()
filtered_df = df.copy()

if time_filter == "Last Week":
    filtered_df = df[df['date'] >= pd.Timestamp(today - timedelta(days=7))]
elif time_filter == "Last Month":
    filtered_df = df[df['date'] >= pd.Timestamp(today - timedelta(days=30))]
elif time_filter == "Last Year":
    filtered_df = df[df['date'] >= pd.Timestamp(today - timedelta(days=365))]

if filtered_df.empty:
    st.warning(f"No data for selected period: {time_filter}")
    st.stop()

# Brand colors
custom_colors = {
    "Amazon": "#064adc",
    "Flipkart": "#fff033",
    "Amazon (Predicted)": "#00FFFF",  # Cyan for predicted Amazon prices
    "Flipkart (Predicted)": "#32CD32"  # Lime Green for predicted Flipkart prices
}

# Price Prediction using Linear Regression
def predict_price(df, retailer):
    # Filter data by retailer
    retailer_data = df[df["source"] == retailer]
    
    # Ensure date is in the correct format
    retailer_data["days_since_start"] = (retailer_data["date"] - retailer_data["date"].min()).dt.days

    # Prepare features and target
    X = retailer_data["days_since_start"].values.reshape(-1, 1)
    y = retailer_data["current_price"].values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices (next 30 days)
    future_dates = np.array([retailer_data["days_since_start"].max() + i for i in range(1, 31)]).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)

    # Create a DataFrame for the predictions
    future_dates = retailer_data["date"].max() + pd.to_timedelta(future_dates.flatten(), unit='D')
    predicted_df = pd.DataFrame({
        "date": future_dates,
        "current_price": predicted_prices,
        "source": retailer,
        "type": "Predicted"
    })
    return predicted_df

# Add predicted prices to the DataFrame
amazon_predicted = predict_price(filtered_df, "Amazon")
flipkart_predicted = predict_price(filtered_df, "Flipkart")

# Combine the actual and predicted data
combined_df = pd.concat([filtered_df, amazon_predicted, flipkart_predicted])

# Bar chart for price comparison (only actual prices)
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:
    st.markdown("<h3 style='text-align: center;'>Price Comparison: Amazon vs Flipkart</h3>", unsafe_allow_html=True)
    bar_fig = px.bar(
        filtered_df,
        x="date",
        y="current_price",
        color="source",
        barmode="group",
        labels={"current_price": "Current Price (₹)", "date": "Date"},
        template="plotly_dark",
        height=500,
        color_discrete_map=custom_colors
    )
    bar_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(tickangle=-45),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# Price stats + latest
col_stats, col_latest = st.columns(2)

with col_stats:
    st.subheader("Price Statistics")
    stats = combined_df.groupby("source")["current_price"].agg(['min', 'max', 'mean']).reset_index()
    stats['mean'] = stats['mean'].round(2)
    stats.columns = ["Source", "Minimum Price (₹)", "Maximum Price (₹)", "Average Price (₹)"]
    st.dataframe(stats, hide_index=True, use_container_width=True)

with col_latest:
    st.subheader("Recent Supplier Wise Price")
    latest_date = combined_df["date"].max()
    latest_prices = combined_df[combined_df["date"] == latest_date][["source", "current_price"]]
    latest_prices.columns = ["Source", "Current Price (₹)"]
    st.dataframe(latest_prices, hide_index=True, use_container_width=True)

# Line chart for price trends (including actual and predicted prices)
st.subheader("Price Trend Over Time (Including Predictions)")
line_fig = px.line(
    combined_df,
    x="date",
    y="current_price",
    color="source",
    labels={"current_price": "Price (₹)", "date": "Date"},
    template="plotly_dark",
    color_discrete_map=custom_colors
)
line_fig.update_layout(
    title="Price Trends (Including 30-Day Prediction)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(line_fig, use_container_width=True)

# Show predicted prices table
st.subheader("Predicted Prices for Next 30 Days")

predicted_table = pd.concat([amazon_predicted, flipkart_predicted])
predicted_table['date'] = predicted_table['date'].dt.strftime("%d-%b-%Y")
predicted_table['current_price'] = predicted_table['current_price'].round(2)
predicted_table.columns = ['Date', 'Predicted Price (₹)', 'Source', 'Type']

st.dataframe(predicted_table, hide_index=True, use_container_width=True)
