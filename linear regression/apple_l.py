import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pymysql
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os
import json

class PricePrediction:
    def __init__(self, db_config=None):
        """
        Initialize the PricePrediction class.
        
        Parameters:
        -----------
        db_config : dict, optional
            Database configuration with keys: 'host', 'user', 'password', 'database'
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'user': 'root',
            'password': 'root',
            'database': 'scraper_db'
        }
        
    def connect_to_db(self):
        """Connect to the MySQL database."""
        try:
            connection = pymysql.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            return connection
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def get_product_data(self, product_name):
        """
        Fetch price data for a specific product.
        
        Parameters:
        -----------
        product_name : str
            Name of the product to fetch data for
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing price data
        """
        # Map product names to view names
        product_view_map = {
            "iphone15": "view_apple_iphone_15_128gb",
            "iphone15pro": "view_apple_iphone_15_pro_128gb",
            "galaxys24": "view_samsung_galaxy_s24_128gb",
            # Add more products as needed
        }
        
        view_name = product_view_map.get(product_name.lower().replace(" ", ""))
        if not view_name:
            print(f"Error: Product '{product_name}' not found in the database")
            return None
        
        conn = self.connect_to_db()
        if not conn:
            return None
        
        try:
            query = f"SELECT * FROM {view_name}"
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Process the data
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            if conn:
                conn.close()
            return None
    
    def predict_arima(self, df, source, days=30, order=(1,1,1)):
        """
        Predict future prices using ARIMA model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        source : str
            Source name (e.g., 'Amazon', 'Flipkart')
        days : int, optional
            Number of days to predict, default is 30
        order : tuple, optional
            ARIMA model order (p,d,q), default is (1,1,1)
            
        Returns:
        --------
        tuple
            (forecast_df, error_metrics)
        """
        try:
            # Filter data for the source
            source_data = df[df['source'] == source].copy()
            
            if len(source_data) < 5:
                print(f"Insufficient data for ARIMA prediction for {source}. Minimum 5 data points required.")
                return None, None
            
            # Prepare time series data
            price_series = source_data.set_index('date')['current_price']
            
            # Split data for training and testing
            train_size = int(len(price_series) * 0.8)
            train, test = price_series[:train_size], price_series[train_size:]
            
            # Train the model
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            
            # Get error metrics if we have test data
            error_metrics = None
            if len(test) > 0:
                test_predictions = model_fit.forecast(steps=len(test))
                rmse = np.sqrt(mean_squared_error(test, test_predictions))
                mae = mean_absolute_error(test, test_predictions)
                error_metrics = {
                    'rmse': rmse,
                    'mae': mae
                }
            
            # Retrain on full dataset for future predictions
            full_model = ARIMA(price_series, order=order)
            full_model_fit = full_model.fit()
            
            # Make forecasts
            forecast = full_model_fit.forecast(steps=days)
            
            # Create date range for forecast
            last_date = source_data['date'].max()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted_price': forecast.values,
                'source': source,
                'model': 'ARIMA'
            })
            
            return forecast_df, error_metrics
            
        except Exception as e:
            print(f"Error in ARIMA prediction for {source}: {e}")
            return None, None
    
    def predict_linear(self, df, source, days=30):
        """
        Predict future prices using Linear Regression.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
        source : str
            Source name (e.g., 'Amazon', 'Flipkart')
        days : int, optional
            Number of days to predict, default is 30
            
        Returns:
        --------
        tuple
            (forecast_df, error_metrics)
        """
        try:
            # Filter data for the source
            source_data = df[df['source'] == source].copy()
            
            if len(source_data) < 2:
                print(f"Insufficient data for Linear Regression prediction for {source}. Minimum 2 data points required.")
                return None, None
            
            # Create features (days since first date)
            first_date = source_data['date'].min()
            source_data['days_since_first'] = (source_data['date'] - first_date).dt.days
            
            # Split data for training and testing
            train_size = int(len(source_data) * 0.8)
            train = source_data[:train_size]
            test = source_data[train_size:]
            
            # Train the model
            X_train = train[['days_since_first']].values
            y_train = train['current_price'].values
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Get error metrics if we have test data
            error_metrics = None
            if len(test) > 0:
                X_test = test[['days_since_first']].values
                y_test = test['current_price'].values
                
                test_predictions = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                mae = mean_absolute_error(y_test, test_predictions)
                error_metrics = {
                    'rmse': rmse,
                    'mae': mae
                }
            
            # Generate dates for prediction
            last_date = source_data['date'].max()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
            days_to_predict = [(date - first_date).days for date in forecast_dates]
            
            # Make predictions
            predictions = model.predict(np.array(days_to_predict).reshape(-1, 1))
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted_price': predictions,
                'source': source,
                'model': 'Linear Regression'
            })
            
            return forecast_df, error_metrics
            
        except Exception as e:
            print(f"Error in Linear Regression prediction for {source}: {e}")
            return None, None
    
    def predict_prices(self, product_name, days=30, model_type='both'):
        """
        Predict prices for a product using specified model.
        
        Parameters:
        -----------
        product_name : str
            Name of the product to predict prices for
        days : int, optional
            Number of days to predict, default is 30
        model_type : str, optional
            Type of model to use: 'arima', 'linear', or 'both'
            
        Returns:
        --------
        dict
            Dictionary with prediction results
        """
        # Get product data
        df = self.get_product_data(product_name)
        if df is None or df.empty:
            return {
                'success': False,
                'message': f"No data available for product: {product_name}"
            }
        
        # Get unique sources
        sources = df['source'].unique()
        
        # Results dictionary
        results = {
            'product': product_name,
            'prediction_days': days,
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sources': {}
        }
        
        all_predictions = []
        
        # Make predictions for each source
        for source in sources:
            results['sources'][source] = {}
            
            # ARIMA prediction
            if model_type.lower() in ['arima', 'both']:
                arima_forecast, arima_metrics = self.predict_arima(df, source, days)
                if arima_forecast is not None:
                    results['sources'][source]['arima'] = {
                        'forecast': arima_forecast.to_dict('records'),
                        'metrics': arima_metrics
                    }
                    all_predictions.append(arima_forecast)
            
            # Linear Regression prediction
            if model_type.lower() in ['linear', 'both']:
                linear_forecast, linear_metrics = self.predict_linear(df, source, days)
                if linear_forecast is not None:
                    results['sources'][source]['linear'] = {
                        'forecast': linear_forecast.to_dict('records'),
                        'metrics': linear_metrics
                    }
                    all_predictions.append(linear_forecast)
        
        results['success'] = True
        
        # If we have predictions, create and save visualizations
        if all_predictions:
            all_predictions_df = pd.concat(all_predictions)
            self._save_prediction_visuals(df, all_predictions_df, product_name)
        
        return results
    
    def _save_prediction_visuals(self, actual_df, predictions_df, product_name):
        """
        Create and save visualizations of predictions.
        
        Parameters:
        -----------
        actual_df : pandas.DataFrame
            DataFrame containing actual price data
        predictions_df : pandas.DataFrame
            DataFrame containing predicted price data
        product_name : str
            Name of the product
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = 'prediction_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Set up the plot
            plt.figure(figsize=(12, 8))
            
            # Plot actual data
            for source in actual_df['source'].unique():
                source_data = actual_df[actual_df['source'] == source]
                plt.plot(source_data['date'], source_data['current_price'], 
                         marker='o', label=f"Actual {source}")
            
            # Plot predictions
            for source in predictions_df['source'].unique():
                for model in predictions_df['model'].unique():
                    model_data = predictions_df[(predictions_df['source'] == source) & 
                                              (predictions_df['model'] == model)]
                    plt.plot(model_data['date'], model_data['predicted_price'], 
                             linestyle='--', marker='x', label=f"{source} {model} Prediction")
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Price (₹)')
            plt.title(f'Price Prediction for {product_name}')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            filename = f"{output_dir}/{product_name}_prediction_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename)
            plt.close()
            
            print(f"Prediction visualization saved to {filename}")
            
        except Exception as e:
            print(f"Error saving prediction visuals: {e}")
    
    def save_prediction_results(self, results, format='json'):
        """
        Save prediction results to a file.
        
        Parameters:
        -----------
        results : dict
            Prediction results
        format : str, optional
            Output format ('json' or 'csv'), default is 'json'
            
        Returns:
        --------
        str
            Path to the saved file
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = 'prediction_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            product_name = results.get('product', 'product')
            date_str = datetime.now().strftime('%Y%m%d')
            
            if format.lower() == 'json':
                # Save as JSON
                filename = f"{output_dir}/{product_name}_prediction_{date_str}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
                
            elif format.lower() == 'csv':
                # Save as CSV
                filename = f"{output_dir}/{product_name}_prediction_{date_str}.csv"
                
                # Convert prediction results to DataFrame
                rows = []
                for source, models in results['sources'].items():
                    for model_type, data in models.items():
                        if 'forecast' in data:
                            for pred in data['forecast']:
                                rows.append({
                                    'product': product_name,
                                    'source': source,
                                    'model': model_type,
                                    'date': pred['date'],
                                    'predicted_price': pred['predicted_price']
                                })
                
                if rows:
                    pd.DataFrame(rows).to_csv(filename, index=False)
                else:
                    print("No prediction data to save.")
                    return None
            else:
                print(f"Unsupported format: {format}")
                return None
            
            print(f"Prediction results saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving prediction results: {e}")
            return None


def main():
    """Main function to run from command line."""
    parser = argparse.ArgumentParser(description='Price Prediction Tool')
    parser.add_argument('product', help='Product name (e.g., iphone15, galaxys24)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to predict (default: 30)')
    parser.add_argument('--model', choices=['arima', 'linear', 'both'], default='both',
                        help='Prediction model to use (default: both)')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                        help='Output format (default: json)')
    
    args = parser.parse_args()
    
    predictor = PricePrediction()
    results = predictor.predict_prices(args.product, args.days, args.model)
    
    if results['success']:
        predictor.save_prediction_results(results, args.format)
    else:
        print(f"Prediction failed: {results.get('message')}")


if __name__ == "__main__":
    main()