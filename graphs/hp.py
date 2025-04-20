import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import pymysql
from datetime import timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# Connect to Cloud (Railway) MySQL database
try:
    connection = pymysql.connect(
        host="nozomi.proxy.rlwy.net",
        user="root",
        password="hkXIIviYSdzhcwvbDyLtyDBDkdnvhLGE",
        port=10649,
        database="railway"
    )

except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# Page layout settings
st.set_page_config(layout="wide", page_title="HP Victus Gaming Laptop Price Tracker")

# Styling the title and last updated section
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
    .stSelectbox label {
        font-weight: bold;
    }
    </style>
    <center><h1 class="title-test">HP VICTUS GAMING LAPTOP PRICE COMPARISON</h1></center>
"""
st.markdown(html_title, unsafe_allow_html=True)

# Display last updated date
box_date = datetime.datetime.now().strftime("%d %B %Y")
st.markdown(f"<p class='update-text'><b>Last updated on:</b> {box_date}</p>", unsafe_allow_html=True)

# Filter selection
filter_col1, filter_col2, filter_col3 = st.columns([0.7, 0.1, 0.2])
with filter_col3:
    time_filter = st.selectbox(
        "Time Period",
        ["All Time", "Last Week", "Last Month", "Last Year"],
        index=0
    )

# Read data from MySQL
try:
    query = "SELECT * FROM view_hp_victus_gaming_laptop"
    df = pd.read_sql(query, connection)
    connection.close()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    if 'connection' in locals() and connection.open:
        connection.close()
    st.stop()

# Check if data is empty
if df.empty:
    st.warning("No data available for HP Victus Gaming Laptop.")
    st.stop()

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])
today = datetime.datetime.now().date()
filtered_df = df.copy()

# Apply time filter
if time_filter == "Last Week":
    filtered_df = df[df['date'] >= pd.Timestamp(today - timedelta(days=7))]
elif time_filter == "Last Month":
    filtered_df = df[df['date'] >= pd.Timestamp(today - timedelta(days=30))]
elif time_filter == "Last Year":
    filtered_df = df[df['date'] >= pd.Timestamp(today - timedelta(days=365))]

if filtered_df.empty:
    st.warning(f"No data available for the selected time period: {time_filter}")
    st.stop()

# Custom colors
custom_colors = {
    "Amazon": "#064adc",
    "Flipkart": "#fff033",
    "Amazon (Predicted)": "#00FFFF",  # Cyan for predicted Amazon prices
    "Flipkart (Predicted)": "#32CD32"  # Lime Green for predicted Flipkart prices
}

# Bar chart for price comparison
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:
    st.markdown("<h3 style='text-align: center;'>Price Comparison: Amazon vs Flipkart</h3>", unsafe_allow_html=True)
    fig = px.bar(filtered_df,
                 x="date",
                 y="current_price",
                 color="source",
                 barmode="group",
                 labels={"current_price": "Current Price (₹)", "date": "Date", "source": "Retailer"},
                 template="plotly_dark",
                 height=500,
                 color_discrete_map=custom_colors)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig, use_container_width=True)

# Stats and latest prices
stats_col1, stats_col2 = st.columns(2)

with stats_col1:
    st.subheader("Price Statistics")
    price_stats = filtered_df.groupby('source')['current_price'].agg(['min', 'max', 'mean']).reset_index()
    price_stats['mean'] = price_stats['mean'].round(2)
    price_stats.columns = ['Source', 'Minimum Price (₹)', 'Maximum Price (₹)', 'Average Price (₹)']
    st.dataframe(price_stats, hide_index=True, use_container_width=True)

with stats_col2:
    st.subheader("Recent Supplier Wise Price")
    latest_date = filtered_df["date"].max()
    recent_prices = filtered_df[filtered_df["date"] == latest_date][["source", "current_price"]]
    recent_prices.columns = ['Source', 'Current Price (₹)']
    st.dataframe(recent_prices, hide_index=True, use_container_width=True)

# Predict next 4 days using linear regression
# Add date ordinal for linear regression
prediction_df = filtered_df.copy()
prediction_df['date_ordinal'] = prediction_df['date'].map(datetime.datetime.toordinal)
predicted_dfs = []

for source in prediction_df['source'].unique():
    source_data = prediction_df[prediction_df['source'] == source]
    
    # Linear regression setup
    X = source_data[['date_ordinal']]
    y = source_data['current_price']
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 4 days
    last_date = source_data['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 5)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predicted_prices = model.predict(future_ordinals)
    
    # Prepare predicted data
    future_df = pd.DataFrame({
        'date': future_dates,
        'current_price': predicted_prices,
        'source': source + ' (Predicted)'
    })
    
    predicted_dfs.append(future_df)

# Combine predictions with actual data
predicted_all = pd.concat(predicted_dfs)
full_df = pd.concat([filtered_df, predicted_all])

# Line chart with predictions (single combined chart for actual and predicted)
st.subheader("Price Trends Over Time (Including Predictions)")
line_fig_pred = px.line(
    full_df, 
    x="date", 
    y="current_price", 
    color="source",
    labels={"current_price": "Price (₹)", "date": "Date", "source": "Retailer"},
    color_discrete_map=custom_colors
)

line_fig_pred.update_layout(
    title="Price Trends (Including 4-Day Prediction)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(line_fig_pred, use_container_width=True)

# Show predicted prices table
st.subheader("Predicted Prices for Next 4 Days")

predicted_table = predicted_all.copy()
predicted_table['date'] = predicted_table['date'].dt.strftime("%d-%b-%Y")
predicted_table['current_price'] = predicted_table['current_price'].round(2)
predicted_table.columns = ['Date', 'Predicted Price (₹)', 'Source']

st.dataframe(predicted_table, hide_index=True, use_container_width=True)
