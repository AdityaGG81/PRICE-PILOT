import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import pymysql
from datetime import timedelta

# Connect to MySQL database
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
st.set_page_config(layout="wide", page_title="Sony Headphone Price Tracker")

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
    <center><h1 class="title-test">SONY HEADPHONE PRICE COMPARISON</h1></center>
"""
st.markdown(html_title, unsafe_allow_html=True)

# Display last updated date in the center
box_date = datetime.datetime.now().strftime("%d %B %Y")
st.markdown(f"<p class='update-text'><b>Last updated on:</b> {box_date}</p>", unsafe_allow_html=True)

# Create a row with columns for filter placement
filter_col1, filter_col2, filter_col3 = st.columns([0.7, 0.1, 0.2])

# Add the filter in the right column
with filter_col3:
    time_filter = st.selectbox(
        "Time Period",
        ["All Time", "Last Week", "Last Month", "Last Year"],
        index=0
    )

# Read data from MySQL
try:
    query = "SELECT * FROM view_sony_headphone"
    df = pd.read_sql(query, connection)
    
    # Close database connection after fetching data
    connection.close()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    if 'connection' in locals() and connection.open:
        connection.close()
    st.stop()

# Check if the dataframe is empty
if df.empty:
    st.warning("No data available for Sony headphones.")
    st.stop()

# Convert date column to datetime if it's not already
df['date'] = pd.to_datetime(df['date'])

# Apply filter based on selection
today = datetime.datetime.now().date()
filtered_df = df.copy()

if time_filter == "Last Week":
    start_date = today - timedelta(days=7)
    filtered_df = df[df['date'] >= pd.Timestamp(start_date)]
elif time_filter == "Last Month":
    start_date = today - timedelta(days=30)
    filtered_df = df[df['date'] >= pd.Timestamp(start_date)]
elif time_filter == "Last Year":
    start_date = today - timedelta(days=365)
    filtered_df = df[df['date'] >= pd.Timestamp(start_date)]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning(f"No data available for the selected time period: {time_filter}")
    st.stop()

# Define custom colors for Amazon and Flipkart
custom_colors = {
    "Amazon": "#064adc",  # Blue (Amazon's brand color)
    "Flipkart": "#fff033"  # Yellow (Flipkart's brand color)
}

# Center-align the chart and title using Streamlit's columns
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])

with col2:
    st.markdown("<h3 style='text-align: center;'>Price Comparison: Amazon vs Flipkart</h3>", unsafe_allow_html=True)
    
    # Create the bar chart
    fig = px.bar(filtered_df,
              x="date",
              y="current_price",
              color="source",
              barmode="group",
              labels={"current_price": "Current Price (₹)", "date": "Date", "source": "Retailer"},
              template="plotly_dark",
              height=500,
              color_discrete_map=custom_colors)
    
    # Customize the figure
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(tickangle=-45)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

# Display statistical information
stats_col1, stats_col2 = st.columns(2)

with stats_col1:
    # Show price statistics
    st.subheader("Price Statistics")
    
    # Group by source and calculate statistics
    price_stats = filtered_df.groupby('source')['current_price'].agg(['min', 'max', 'mean']).reset_index()
    price_stats['mean'] = price_stats['mean'].round(2)
    price_stats.columns = ['Source', 'Minimum Price (₹)', 'Maximum Price (₹)', 'Average Price (₹)']
    st.dataframe(price_stats, hide_index=True, use_container_width=True)

with stats_col2:
    # Display recent prices in the expander
    st.subheader("Recent Supplier Wise Price")
    
    # Get the most recent date in the filtered dataset
    latest_date = filtered_df["date"].max()
    recent_prices = filtered_df[filtered_df["date"] == latest_date][["source", "current_price"]]
    
    # Format the dataframe
    recent_prices.columns = ['Source', 'Current Price (₹)']
    
    # Show the latest prices
    st.dataframe(recent_prices, hide_index=True, use_container_width=True)

# Add a price prediction section (you can add your prediction logic here)
st.subheader("Price Predictions")

# Prepare data for prediction
prediction_df = filtered_df.copy()
prediction_df['date_ordinal'] = prediction_df['date'].map(datetime.datetime.toordinal)

# List to hold predicted dataframes
predicted_dfs = []

# Perform linear regression for each source (Amazon, Flipkart)
for source in prediction_df['source'].unique():
    source_data = prediction_df[prediction_df['source'] == source]
    
    # Linear regression setup
    from sklearn.linear_model import LinearRegression
    import numpy as np
    X = source_data[['date_ordinal']]  # Independent variable (dates converted to ordinal)
    y = source_data['current_price']   # Dependent variable (prices)
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the next 4 days' prices
    last_date = source_data['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 5)]  # Next 4 days
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)  # Convert future dates to ordinal
    predicted_prices = model.predict(future_ordinals)  # Get predictions
    
    # Create a DataFrame for predicted prices
    future_df = pd.DataFrame({
        'date': future_dates,
        'current_price': predicted_prices,
        'source': f"{source} (Predicted)"  # Add '(Predicted)' to source name
    })
    
    # Append the predicted dataframe to the list
    predicted_dfs.append(future_df)

# Combine predicted data with actual data
predicted_all = pd.concat(predicted_dfs)
full_df = pd.concat([filtered_df, predicted_all])

# Line chart showing price trends (including predicted data)
line_fig = px.line(
    full_df, 
    x="date", 
    y="current_price", 
    color="source",
    labels={"current_price": "Price (₹)", "date": "Date", "source": "Retailer"},
    color_discrete_map={
        "Amazon": "#064adc",  # Amazon brand color
        "Flipkart": "#fff033",  # Flipkart brand color
        "Amazon (Predicted)":"#00FFFF",  # Green for predicted Amazon prices
        "Flipkart (Predicted)": "#32CD32"  # Orange for predicted Flipkart prices
    }
)

line_fig.update_layout(
    title="Price Trends (Including 4-Day Prediction)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=40, b=40)
)

# Display the plot
st.plotly_chart(line_fig, use_container_width=True)

# Show predicted prices table for the next 4 days
st.subheader("Predicted Prices for Next 4 Days")

predicted_table = predicted_all.copy()
predicted_table['date'] = predicted_table['date'].dt.strftime("%d-%b-%Y")  # Format date for better readability
predicted_table['current_price'] = predicted_table['current_price'].round(2)  # Round prices to 2 decimal places
predicted_table.columns = ['Date', 'Predicted Price (₹)', 'Source']  # Rename columns for clarity

# Display the table
st.dataframe(predicted_table, hide_index=True, use_container_width=True)
