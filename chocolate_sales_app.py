import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🍫 Chocolate Sales Forecasting",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #8B4513;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #8B4513;
    margin: 0.5rem 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #f0f2f6;
    border-radius: 4px 4px 0px 0px;
}
.stTabs [aria-selected="true"] {
    background-color: #8B4513;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍫 Chocolate Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Navigation")
st.sidebar.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Chocolate Sales CSV", 
    type=['csv'],
    help="Upload your chocolate sales data file"
)

# Global variables for data processing
@st.cache_data
def load_and_process_data(file):
    """Load and process the chocolate sales data"""
    df = pd.read_csv(file)
    
    # Data cleaning
    df['Amount'] = df['Amount'].replace('[\$]', '', regex=True)
    df['Amount'] = df['Amount'].replace(',', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.set_index('Date', inplace=True)
    
    # Group by Product and Date
    grouped_date = df.groupby(['Product', 'Date'])[['Amount', 'Boxes Shipped']].sum().reset_index()
    grouped_date = grouped_date.sort_values(by=['Product', 'Date']).reset_index(drop=True)
    
    # Resample to weekly data
    grouped_date['Date'] = pd.to_datetime(grouped_date['Date'])
    grouped_date = grouped_date.set_index('Date')
    weekly = (grouped_date.groupby('Product')
              .resample('W-MON')[['Amount', 'Boxes Shipped']]
              .sum()
              .reset_index())
    weekly = weekly.sort_values(by=['Product', 'Date']).reset_index(drop=True)
    
    return df, weekly

def preprocess_for_lstm(weekly_data):
    """Preprocess data for LSTM modeling"""
    weekly_small = weekly_data[['Date', 'Product', 'Amount']].copy()
    weekly_small['Amount'] = weekly_small['Amount'].replace(0, np.nan)
    
    # Outlier treatment using IQR
    def remove_outliers_iqr(group):
        Q1 = group['Amount'].quantile(0.25)
        Q3 = group['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        group['Amount'] = np.where(group['Amount'] < lower_bound, lower_bound, group['Amount'])
        group['Amount'] = np.where(group['Amount'] > upper_bound, upper_bound, group['Amount'])
        return group
    
    weekly_small = weekly_small.groupby('Product').apply(remove_outliers_iqr).reset_index(drop=True)
    
    # Smoothing
    def smooth_amount(group, window=3):
        group['Amount'] = group['Amount'].rolling(window, min_periods=1, center=True).mean()
        return group
    
    weekly_small = weekly_small.groupby('Product').apply(smooth_amount).reset_index(drop=True)
    
    # Interpolation
    weekly_small = weekly_small.sort_values(['Product', 'Date']).reset_index(drop=True)
    def interpolate_amount(group):
        group['Amount'] = group['Amount'].interpolate(method='linear', limit_direction='both')
        return group
    
    interpolated = weekly_small.groupby('Product').apply(interpolate_amount).reset_index(drop=True)
    
    return interpolated

def create_sequence(data, seq_length):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Main app logic
if uploaded_file is not None:
    # Load and process data
    with st.spinner('Loading and processing data...'):
        df, weekly_data = load_and_process_data(uploaded_file)
        interpolated_data = preprocess_for_lstm(weekly_data)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Data Overview", "🔍 EDA", "🤖 Model Training", "📊 Results", "🔮 Forecasting"])
    
    with tab1:
        st.header("📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Products", df['Product'].nunique())
        with col3:
            st.metric("Date Range", f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Total Sales", f"${df['Amount'].sum():,.0f}")
        
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(10))
        
        st.subheader("Weekly Aggregated Data")
        st.dataframe(weekly_data.head(10))
        
        st.subheader("Data Quality Check")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Missing Values:**")
            missing_data = df.isnull().sum()
            st.write(missing_data[missing_data > 0])
        
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes)
    
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        # Sales by Product
        product_sales = df.groupby('Product')['Amount'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=product_sales.index, 
            y=product_sales.values,
            title="Total Sales by Product",
            labels={'x': 'Product', 'y': 'Total Sales ($)'},
            color=product_sales.values,
            color_continuous_scale='Browns'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series plot
        selected_products = st.multiselect(
            "Select products for time series analysis:",
            options=weekly_data['Product'].unique(),
            default=weekly_data['Product'].unique()[:3]
        )
        
        if selected_products:
            fig = go.Figure()
            for product in selected_products:
                product_data = weekly_data[weekly_data['Product'] == product]
                fig.add_trace(go.Scatter(
                    x=product_data['Date'],
                    y=product_data['Amount'],
                    mode='lines+markers',
                    name=product,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Weekly Sales Trends by Product",
                xaxis_title="Date",
                yaxis_title="Sales Amount ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot for sales distribution
        fig = px.box(
            weekly_data, 
            x='Product', 
            y='Amount',
            title="Sales Distribution by Product"
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🤖 Model Training")
        
        # Model parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            seq_length = st.slider("Sequence Length", 2, 10, 4)
        with col2:
            lstm_units = st.slider("LSTM Units", 32, 256, 128)
        with col3:
            epochs = st.slider("Max Epochs", 50, 200, 100)
        
        if st.button("🚀 Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training results storage
            results = []
            products = interpolated_data['Product'].unique()
            
            for idx, product in enumerate(products):
                status_text.text(f'Training model for {product}...')
                progress_bar.progress((idx + 1) / len(products))
                
                product_df = interpolated_data[interpolated_data['Product'] == product].sort_values(by='Date')
                product_df.set_index('Date', inplace=True)
                
                if len(product_df) < seq_length + 3:
                    continue
                
                # Normalize data
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
                
                # Build model
                model = Sequential([
                    LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dropout(0.1),
                    LSTM(90),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                # Train model
                history = model.fit(
                    X_train, y_train, 
                    epochs=epochs, 
                    batch_size=8, 
                    verbose=0,
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stop]
                )
                
                # Make predictions
                pred_scaled = model.predict(X_test, verbose=0)
                predicted_amounts = scaler.inverse_transform(pred_scaled).flatten()
                actual_amounts = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                mape = mean_absolute_percentage_error(actual_amounts, predicted_amounts)
                mae = mean_absolute_error(actual_amounts, predicted_amounts)
                rmse = np.sqrt(mean_squared_error(actual_amounts, predicted_amounts))
                r2 = r2_score(actual_amounts, predicted_amounts)
                
                results.append({
                    'Product': product,
                    'MAPE': mape,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Actual_Avg': np.mean(actual_amounts),
                    'Predicted_Avg': np.mean(predicted_amounts),
                    'Best_Epoch': np.argmin(history.history['val_loss']) + 1
                })
            
            # Store results in session state
            st.session_state['training_results'] = pd.DataFrame(results)
            st.session_state['model_trained'] = True
            
            status_text.text('Training completed!')
            st.success("✅ Model training completed successfully!")
    
    with tab4:
        st.header("📊 Model Results")
        
        if 'training_results' in st.session_state:
            results_df = st.session_state['training_results']
            
            # Overall metrics
            st.subheader("📈 Overall Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_mape = results_df['MAPE'].mean()
                st.metric("Average MAPE", f"{avg_mape:.1f}%")
            with col2:
                avg_mae = results_df['MAE'].mean()
                st.metric("Average MAE", f"${avg_mae:,.0f}")
            with col3:
                avg_rmse = results_df['RMSE'].mean()
                st.metric("Average RMSE", f"${avg_rmse:,.0f}")
            with col4:
                avg_r2 = results_df['R2'].mean()
                st.metric("Average R²", f"{avg_r2:.3f}")
            
            # Results table
            st.subheader("📋 Detailed Results by Product")
            display_df = results_df[['Product', 'MAPE', 'MAE', 'RMSE', 'R2', 'Best_Epoch']].round(2)
            st.dataframe(display_df, use_container_width=True)
            
            # Performance visualization
            st.subheader("📊 Model Performance Visualization")
            
            # MAPE by product
            fig = px.bar(
                results_df.sort_values('MAPE'), 
                x='Product', 
                y='MAPE',
                title="MAPE by Product",
                color='MAPE',
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Actual vs Predicted
            fig = px.scatter(
                results_df, 
                x='Actual_Avg', 
                y='Predicted_Avg',
                hover_data=['Product'],
                title="Actual vs Predicted Weekly Averages",
                labels={'Actual_Avg': 'Actual Average ($)', 'Predicted_Avg': 'Predicted Average ($)'}
            )
            
            # Add diagonal line
            min_val = min(results_df['Actual_Avg'].min(), results_df['Predicted_Avg'].min())
            max_val = max(results_df['Actual_Avg'].max(), results_df['Predicted_Avg'].max())
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash")
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Please train the model first in the 'Model Training' tab to see results.")
    
    with tab5:
        st.header("🔮 Sales Forecasting")
        
        if 'model_trained' in st.session_state:
            st.subheader("📅 Generate Forecasts")
            
            col1, col2 = st.columns(2)
            with col1:
                forecast_horizon = st.slider("Forecast Horizon (weeks)", 1, 12, 4)
            with col2:
                selected_product = st.selectbox(
                    "Select Product for Detailed Forecast",
                    options=interpolated_data['Product'].unique()
                )
            
            if st.button("🔮 Generate Forecasts", type="primary"):
                with st.spinner('Generating forecasts...'):
                    forecast_results = []
                    products = interpolated_data['Product'].unique()
                    
                    for product in products:
                        product_df = interpolated_data[interpolated_data['Product'] == product].sort_values('Date')
                        product_df.set_index('Date', inplace=True)
                        
                        if len(product_df) < seq_length + forecast_horizon:
                            continue
                        
                        # Scale data
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(product_df[['Amount']])
                        
                        # Create sequences for training
                        X, y = [], []
                        for i in range(len(scaled_data) - seq_length):
                            X.append(scaled_data[i:i+seq_length])
                            y.append(scaled_data[i+seq_length])
                        X, y = np.array(X), np.array(y)
                        
                        # Build and train model
                        model = Sequential([
                            LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
                            Dropout(0.1),
                            LSTM(90),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        model.fit(X, y, epochs=100, batch_size=8, verbose=0)
                        
                        # Generate forecasts
                        input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
                        forecasts = []
                        for _ in range(forecast_horizon):
                            pred_scaled = model.predict(input_seq, verbose=0)
                            pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
                            forecasts.append(pred_value)
                            
                            # Update input sequence
                            next_input = pred_scaled.reshape(1, 1, 1)
                            input_seq = np.concatenate((input_seq[:, 1:, :], next_input), axis=1)
                        
                        forecast_results.append([product] + forecasts)
                    
                    # Create forecast DataFrame
                    columns = ['Product'] + [f'Week+{i+1}' for i in range(forecast_horizon)]
                    forecast_df = pd.DataFrame(forecast_results, columns=columns)
                    
                    st.session_state['forecast_df'] = forecast_df
                    st.success("✅ Forecasts generated successfully!")
            
            # Display forecasts
            if 'forecast_df' in st.session_state:
                forecast_df = st.session_state['forecast_df']
                
                st.subheader("📊 Forecast Results")
                st.dataframe(forecast_df.round(2), use_container_width=True)
                
                # Detailed forecast for selected product
                if selected_product in forecast_df['Product'].values:
                    st.subheader(f"📈 Detailed Forecast for {selected_product}")
                    
                    product_forecast = forecast_df[forecast_df['Product'] == selected_product].iloc[0]
                    forecast_values = [product_forecast[f'Week+{i+1}'] for i in range(forecast_horizon)]
                    
                    # Get historical data
                    historical_data = interpolated_data[
                        interpolated_data['Product'] == selected_product
                    ].tail(12)  # Last 12 weeks
                    
                    # Create forecast visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=historical_data['Date'],
                        y=historical_data['Amount'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast data
                    last_date = historical_data['Date'].max()
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(weeks=1),
                        periods=forecast_horizon,
                        freq='W'
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'Sales Forecast for {selected_product}',
                        xaxis_title='Date',
                        yaxis_title='Sales Amount ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download forecast
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name="chocolate_sales_forecast.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("👆 Please train the model first in the 'Model Training' tab to generate forecasts.")

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Chocolate Sales Forecasting Dashboard! 🍫
    
    This application uses LSTM neural networks to forecast chocolate sales based on historical data.
    
    ### Features:
    - 📈 **Data Overview**: Explore your sales data with comprehensive statistics
    - 🔍 **Exploratory Data Analysis**: Visualize sales trends and patterns
    - 🤖 **Model Training**: Train LSTM models for each product
    - 📊 **Results Analysis**: Evaluate model performance with detailed metrics
    - 🔮 **Sales Forecasting**: Generate future sales predictions
    
    ### Getting Started:
    1. Upload your chocolate sales CSV file using the sidebar
    2. Navigate through the tabs to explore your data
    3. Train the LSTM models
    4. Generate forecasts for future sales
    
    ### Data Format:
    Your CSV should contain columns: `Date`, `Product`, `Amount`, `Boxes Shipped`
    """)
    
    # Sample data format
    st.subheader("Expected Data Format:")
    sample_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Product': ['Dark Chocolate', 'Milk Chocolate', 'White Chocolate'],
        'Amount': ['$1,500', '$2,000', '$1,200'],
        'Boxes Shipped': [150, 200, 120]
    })
    st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🍫 Chocolate Sales Forecasting Dashboard | Built with Streamlit & TensorFlow"
    "</div>", 
    unsafe_allow_html=True
)