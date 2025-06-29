import os
import pandas as pd
import numpy as np
import io
import base64
import json
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store data
interpolated_data = None
forecast_results = None

def create_sample_data():
    """Create sample weekly sales data for demonstration"""
    np.random.seed(42)
    
    # Generate dates for 52 weeks
    dates = pd.date_range(start='2023-01-01', periods=52, freq='W')
    
    # Sample products
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    
    data = []
    for product in products:
        base_amount = np.random.uniform(100, 500)
        trend = np.random.uniform(-2, 2)
        seasonality = np.random.uniform(0.1, 0.3)
        
        for i, date in enumerate(dates):
            # Add trend, seasonality, and noise
            amount = (base_amount + 
                     trend * i + 
                     seasonality * base_amount * np.sin(2 * np.pi * i / 12) +
                     np.random.normal(0, base_amount * 0.1))
            
            # Ensure positive values
            amount = max(amount, 10)
            
            data.append({
                'Date': date,
                'Product': product,
                'Amount': amount
            })
    
    return pd.DataFrame(data)

def remove_outliers_iqr(group):
    Q1 = group['Amount'].quantile(0.25)
    Q3 = group['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group['Amount'] = np.where(group['Amount'] < lower_bound, lower_bound, group['Amount'])
    group['Amount'] = np.where(group['Amount'] > upper_bound, upper_bound, group['Amount'])
    return group

def smooth_amount(group, window=3):
    group['Amount'] = group['Amount'].rolling(window, min_periods=1, center=True).mean()
    return group

def interpolate_amount(group):
    group['Amount'] = group['Amount'].interpolate(method='linear', limit_direction='both')
    return group

def preprocess_data(df):
    """Preprocess the data following your original logic"""
    # 1. Prepare data
    weekly_small = df[['Date', 'Product', 'Amount']].copy()
    weekly_small['Amount'] = weekly_small['Amount'].replace(0, np.nan)
    
    # 2. Outlier treatment
    weekly_small = weekly_small.groupby('Product').apply(remove_outliers_iqr).reset_index(drop=True)
    
    # 3. Smoothing
    weekly_small = weekly_small.groupby('Product').apply(smooth_amount).reset_index(drop=True)
    
    # 4. Interpolation
    weekly_small = weekly_small.sort_values(['Product', 'Date']).reset_index(drop=True)
    interpolated = weekly_small.groupby('Product').apply(interpolate_amount).reset_index(drop=True)
    
    return interpolated

def create_sequence(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def train_and_evaluate_models(data):
    """Train LSTM models and return evaluation results"""
    product_names = []
    actual_dict = {}
    predicted_dict = []
    best_epochs = []
    
    products = data['Product'].unique()
    seq_length = 4
    
    for product in products:
        product_df = data[data['Product'] == product].sort_values(by='Date')
        product_df.set_index('Date', inplace=True)
        
        if len(product_df) < seq_length + 3:
            continue
        
        # Normalize Amount
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(product_df[['Amount']])
        
        # Create sequences
        X, y = create_sequence(scaled_data, seq_length)
        if len(X) == 0:
            continue
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        # LSTM Model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.1))
        model.add(LSTM(90))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train, epochs=100, batch_size=8, verbose=0,
            validation_data=(X_test, y_test), callbacks=[early_stop]
        )
        
        # Record best epoch
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_epochs.append(best_epoch)
        
        # Predict and evaluate
        pred_scaled = model.predict(X_test, verbose=0)
        predicted_amounts = scaler.inverse_transform(pred_scaled).flatten()
        actual_amounts = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Store results
        product_names.append(product)
        actual_dict[product] = np.mean(actual_amounts)
        predicted_dict.append(np.mean(predicted_amounts))
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Product': product_names,
        'Actual Weekly Avg': [actual_dict[p] for p in product_names],
        'Predicted Weekly Avg': [float(p) for p in predicted_dict],
        'Best Epoch': best_epochs
    }).round(1)
    
    return results_df

def generate_forecasts(data):
    """Generate 4-week forecasts for all products"""
    seq_length = 4
    forecast_horizon = 4
    forecast_data = []
    
    products = data['Product'].unique()
    
    for product in products:
        product_df = data[data['Product'] == product].sort_values('Date')
        product_df.set_index('Date', inplace=True)
        
        if len(product_df) < seq_length + forecast_horizon:
            continue
        
        # Scale Amount
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(product_df[['Amount']])
        
        # Create sequences for training
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(scaled_data[i+seq_length])
        X, y = np.array(X), np.array(y)
        
        # LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.1))
        model.add(LSTM(90))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=100, batch_size=8, verbose=0)
        
        # Forecast recursively
        input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        preds = []
        for _ in range(forecast_horizon):
            pred_scaled = model.predict(input_seq, verbose=0)
            pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
            preds.append(pred_value)
            
            # Update input_seq for next prediction
            next_input = pred_scaled.reshape(1, 1, 1)
            input_seq = np.concatenate((input_seq[:, 1:, :], next_input), axis=1)
        
        forecast_data.append([product] + preds)
    
    # Convert to DataFrame
    forecast_df = pd.DataFrame(
        forecast_data, 
        columns=['Product', 'Week+1', 'Week+2', 'Week+3', 'Week+4']
    )
    
    return forecast_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global interpolated_data
    
    if 'file' not in request.files:
        # Use sample data if no file uploaded
        df = create_sample_data()
    else:
        file = request.files['file']
        if file.filename == '':
            df = create_sample_data()
        else:
            df = pd.read_csv(file)
    
    # Preprocess data
    interpolated_data = preprocess_data(df)
    
    return jsonify({
        'success': True,
        'message': 'Data processed successfully',
        'products': list(interpolated_data['Product'].unique()),
        'data_shape': interpolated_data.shape
    })

@app.route('/train')
def train_models():
    global interpolated_data
    
    if interpolated_data is None:
        return jsonify({'error': 'No data available. Please upload data first.'})
    
    # Train models and get evaluation results
    results_df = train_and_evaluate_models(interpolated_data)
    
    # Calculate overall metrics
    actual_means = results_df['Actual Weekly Avg'].values
    predicted_means = results_df['Predicted Weekly Avg'].values
    
    epsilon = 1e-10
    mape_avg = np.mean(np.abs((actual_means - predicted_means) / (actual_means + epsilon))) * 100
    mae_avg = mean_absolute_error(actual_means, predicted_means)
    rmse_avg = np.sqrt(mean_squared_error(actual_means, predicted_means))
    r2_avg = r2_score(actual_means, predicted_means)
    rmse_pct = (rmse_avg / np.mean(actual_means)) * 100
    
    metrics = {
        'MAPE': round(mape_avg, 1),
        'MAE': round(mae_avg, 1),
        'RMSE': round(rmse_avg, 1),
        'RMSE_PCT': round(rmse_pct, 1),
        'R2': round(r2_avg, 3)
    }
    
    return jsonify({
        'success': True,
        'results': results_df.to_dict('records'),
        'metrics': metrics
    })

@app.route('/forecast')
def generate_forecast():
    global interpolated_data, forecast_results
    
    if interpolated_data is None:
        return jsonify({'error': 'No data available. Please upload data first.'})
    
    # Generate forecasts
    forecast_results = generate_forecasts(interpolated_data)
    
    return jsonify({
        'success': True,
        'forecasts': forecast_results.to_dict('records')
    })

@app.route('/plot/forecast')
def plot_forecast():
    global forecast_results
    
    if forecast_results is None:
        return jsonify({'error': 'No forecast data available.'})
    
    # Create interactive Plotly chart
    weeks = ['Week+1', 'Week+2', 'Week+3', 'Week+4']
    fig = go.Figure()
    
    for index, row in forecast_results.iterrows():
        product = row['Product']
        values = row[weeks].values.astype(float)
        fig.add_trace(go.Bar(
            x=weeks,
            y=values,
            name=product
        ))
    
    fig.update_layout(
        title='📦 4-Week Forecast for All Products',
        xaxis_title='Week',
        yaxis_title='Forecasted Amount',
        barmode='group',
        legend_title='Product',
        template='plotly_white',
        width=1000,
        height=600
    )
    
    # Convert to JSON
    graph_json = pio.to_json(fig)
    
    return jsonify({'plot': graph_json})

@app.route('/download/forecast')
def download_forecast():
    global forecast_results
    
    if forecast_results is None:
        return jsonify({'error': 'No forecast data available.'})
    
    # Create CSV in memory
    output = io.StringIO()
    forecast_results.to_csv(output, index=False)
    output.seek(0)
    
    # Convert to BytesIO for sending
    mem = io.BytesIO()
    mem.write(output.getvalue().encode())
    mem.seek(0)
    
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name='forecast_next_4_weeks.csv'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
