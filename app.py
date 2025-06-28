import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LSTM Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 LSTM Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Helper functions
@st.cache_data
def remove_outliers_iqr(group):
    Q1 = group['Amount'].quantile(0.25)
    Q3 = group['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group['Amount'] = np.where(group['Amount'] < lower_bound, lower_bound, group['Amount'])
    group['Amount'] = np.where(group['Amount'] > upper_bound, upper_bound, group['Amount'])
    return group

@st.cache_data
def smooth_amount(group, window=3):
    group['Amount'] = group['Amount'].rolling(window, min_periods=1, center=True).mean()
    return group

@st.cache_data
def interpolate_amount(group):
    group['Amount'] = group['Amount'].interpolate(method='linear', limit_direction='both')
    return group

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

@st.cache_data
def preprocess_data(df):
    """Preprocess the uploaded data"""
    # Select columns & replace 0 with NaN
    weekly_small = df[['Date', 'Product', 'Amount']].copy()
    weekly_small['Date'] = pd.to_datetime(weekly_small['Date'])
    weekly_small['Amount'] = weekly_small['Amount'].replace(0, np.nan)
    
    # Outlier treatment using IQR capping per product
    weekly_small = weekly_small.groupby('Product').apply(remove_outliers_iqr).reset_index(drop=True)
    
    # Smoothing with rolling mean
    weekly_small = weekly_small.groupby('Product').apply(smooth_amount).reset_index(drop=True)
    
    # Interpolation of missing values
    weekly_small = weekly_small.sort_values(['Product', 'Date']).reset_index(drop=True)
    interpolated = weekly_small.groupby('Product').apply(interpolate_amount).reset_index(drop=True)
    
    return interpolated

def train_lstm_model(product_data, seq_length=4):
    """Train LSTM model for a single product"""
    if len(product_data) < seq_length + 3:
        return None, None, None, None
    
    # Normalize Amount
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(product_data[['Amount']])
    
    # Create sequences
    X, y = create_sequence(scaled_data, seq_length)
    if len(X) == 0:
        return None, None, None, None
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    
    # LSTM Model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(90))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0,
                        validation_data=(X_test, y_test), callbacks=[early_stop])
    
    # Predict and evaluate
    pred_scaled = model.predict(X_test, verbose=0)
    predicted_amounts = scaler.inverse_transform(pred_scaled).flatten()
    actual_amounts = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    return model, scaler, history, (actual_amounts, predicted_amounts)

def generate_forecast(model, scaler, last_sequence, forecast_horizon=4):
    """Generate forecast for future periods"""
    input_seq = last_sequence.reshape(1, len(last_sequence), 1)
    preds = []
    
    for _ in range(forecast_horizon):
        pred_scaled = model.predict(input_seq, verbose=0)
        pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
        preds.append(pred_value)
        
        # Update input_seq for next prediction
        next_input = pred_scaled.reshape(1, 1, 1)
        input_seq = np.concatenate((input_seq[:, 1:, :], next_input), axis=1)
    
    return preds

# Sidebar for file upload and parameters
st.sidebar.header("📁 Data Upload & Parameters")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="CSV should contain columns: Date, Product, Amount"
)

seq_length = st.sidebar.slider("Sequence Length", 2, 10, 4, help="Number of time steps to look back")
forecast_horizon = st.sidebar.slider("Forecast Horizon", 1, 12, 4, help="Number of periods to forecast")

if uploaded_file is not None:
    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_cols = ['Date', 'Product', 'Amount']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ CSV must contain columns: {required_cols}")
            st.stop()
        
        interpolated = preprocess_data(df)
    
    st.success("✅ Data loaded and preprocessed successfully!")
    
    # Data overview
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(interpolated))
    with col2:
        st.metric("Number of Products", interpolated['Product'].nunique())
    with col3:
        st.metric("Date Range", f"{interpolated['Date'].min().strftime('%Y-%m-%d')} to {interpolated['Date'].max().strftime('%Y-%m-%d')}")
    with col4:
        st.metric("Average Weekly Sales", f"{interpolated['Amount'].mean():.1f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Visualization", "🤖 Model Training", "🔮 Forecasting", "📋 Results Summary"])
    
    with tab1:
        st.subheader("Weekly Sales Trends by Product")
        
        # Interactive plot with Plotly
        fig = px.line(interpolated, x='Date', y='Amount', color='Product',
                     title='Weekly Sales Trends (All Products)',
                     labels={'Amount': 'Sales Amount', 'Date': 'Date'})
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plot
        st.subheader("Sales Distribution by Product")
        fig2 = px.box(interpolated, x='Product', y='Amount',
                     title='Sales Distribution by Product')
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("🤖 LSTM Model Training")
        
        if st.button("🚀 Train Models", type="primary"):
            products = interpolated['Product'].unique()
            
            # Initialize storage for results
            results_data = []
            training_progress = st.progress(0)
            status_text = st.empty()
            
            for idx, product in enumerate(products):
                status_text.text(f"Training model for {product}...")
                
                product_df = interpolated[interpolated['Product'] == product].sort_values(by='Date')
                product_df.set_index('Date', inplace=True)
                
                model, scaler, history, eval_data = train_lstm_model(product_df, seq_length)
                
                if model is not None:
                    actual_amounts, predicted_amounts = eval_data
                    
                    # Calculate metrics
                    mape = mean_absolute_percentage_error(actual_amounts, predicted_amounts)
                    mae = mean_absolute_error(actual_amounts, predicted_amounts)
                    rmse = np.sqrt(mean_squared_error(actual_amounts, predicted_amounts))
                    r2 = r2_score(actual_amounts, predicted_amounts)
                    
                    results_data.append({
                        'Product': product,
                        'MAPE': mape,
                        'MAE': mae,
                        'RMSE': rmse,
                        'R2_Score': r2,
                        'Actual_Avg': np.mean(actual_amounts),
                        'Predicted_Avg': np.mean(predicted_amounts)
                    })
                    
                    # Store model and scaler in session state
                    if 'models' not in st.session_state:
                        st.session_state.models = {}
                    if 'scalers' not in st.session_state:
                        st.session_state.scalers = {}
                    
                    st.session_state.models[product] = model
                    st.session_state.scalers[product] = scaler
                
                training_progress.progress((idx + 1) / len(products))
            
            status_text.text("✅ Training completed!")
            
            # Store results in session state
            st.session_state.results_df = pd.DataFrame(results_data)
            
            # Display training results
            if len(results_data) > 0:
                st.subheader("📊 Training Results")
                st.dataframe(st.session_state.results_df.round(2), use_container_width=True)
                
                # Overall metrics
                results_df = st.session_state.results_df
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg MAPE", f"{results_df['MAPE'].mean():.1f}%")
                with col2:
                    st.metric("Avg MAE", f"{results_df['MAE'].mean():.1f}")
                with col3:
                    st.metric("Avg RMSE", f"{results_df['RMSE'].mean():.1f}")
                with col4:
                    st.metric("Avg R² Score", f"{results_df['R2_Score'].mean():.3f}")
    
    with tab3:
        st.subheader("🔮 Sales Forecasting")
        
        if 'models' in st.session_state and st.button("📈 Generate Forecasts", type="primary"):
            forecast_data = []
            products = list(st.session_state.models.keys())
            
            forecast_progress = st.progress(0)
            forecast_status = st.empty()
            
            for idx, product in enumerate(products):
                forecast_status.text(f"Generating forecast for {product}...")
                
                product_df = interpolated[interpolated['Product'] == product].sort_values('Date')
                
                if len(product_df) >= seq_length:
                    # Get last sequence for forecasting
                    scaler = st.session_state.scalers[product]
                    last_values = product_df['Amount'].tail(seq_length).values
                    last_sequence_scaled = scaler.transform(last_values.reshape(-1, 1)).flatten()
                    
                    # Generate forecast
                    model = st.session_state.models[product]
                    forecast = generate_forecast(model, scaler, last_sequence_scaled, forecast_horizon)
                    
                    forecast_row = [product] + forecast
                    forecast_data.append(forecast_row)
                
                forecast_progress.progress((idx + 1) / len(products))
            
            forecast_status.text("✅ Forecasting completed!")
            
            # Create forecast DataFrame
            week_cols = [f'Week+{i+1}' for i in range(forecast_horizon)]
            forecast_df = pd.DataFrame(forecast_data, columns=['Product'] + week_cols)
            st.session_state.forecast_df = forecast_df
            
            # Display forecast table
            st.subheader("📋 Forecast Results")
            st.dataframe(forecast_df.round(1), use_container_width=True)
            
            # Interactive forecast visualization
            st.subheader("📊 Interactive Forecast Visualization")
            
            fig = go.Figure()
            
            for index, row in forecast_df.iterrows():
                product = row['Product']
                values = row[week_cols].values.astype(float)
                fig.add_trace(go.Bar(
                    x=week_cols,
                    y=values,
                    name=product,
                    text=[f'{v:.1f}' for v in values],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='📦 Sales Forecast for All Products',
                xaxis_title='Week',
                yaxis_title='Forecasted Amount',
                barmode='group',
                legend_title='Product',
                template='plotly_white',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual product forecasts
            st.subheader("📈 Individual Product Forecasts")
            selected_product = st.selectbox("Select Product", forecast_df['Product'].unique())
            
            if selected_product:
                product_row = forecast_df[forecast_df['Product'] == selected_product].iloc[0]
                values = product_row[week_cols].values.astype(float)
                
                fig_individual = go.Figure()
                fig_individual.add_trace(go.Scatter(
                    x=week_cols,
                    y=values,
                    mode='lines+markers',
                    name=selected_product,
                    line=dict(width=3),
                    marker=dict(size=10)
                ))
                
                fig_individual.update_layout(
                    title=f'📈 4-Week Forecast for {selected_product}',
                    xaxis_title='Week',
                    yaxis_title='Forecasted Amount',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_individual, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Complete Results Summary")
        
        if 'results_df' in st.session_state and 'forecast_df' in st.session_state:
            # Combine training and forecast results
            results_summary = st.session_state.results_df.merge(
                st.session_state.forecast_df, on='Product', how='left'
            )
            
            st.dataframe(results_summary.round(2), use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_results = results_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Download Training Results",
                    data=csv_results,
                    file_name='lstm_training_results.csv',
                    mime='text/csv'
                )
            
            with col2:
                if 'forecast_df' in st.session_state:
                    csv_forecast = st.session_state.forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Results",
                        data=csv_forecast,
                        file_name='lstm_forecast_results.csv',
                        mime='text/csv'
                    )
        else:
            st.info("🔄 Please complete model training and forecasting to view the summary.")

else:
    st.info("👆 Please upload a CSV file to get started!")
    
    # Sample data format
    st.subheader("📝 Expected Data Format")
    sample_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22'],
        'Product': ['Product A', 'Product A', 'Product B', 'Product B'],
        'Amount': [1000, 1200, 800, 950]
    })
    st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("""
    **Required columns:**
    - **Date**: Date in YYYY-MM-DD format
    - **Product**: Product name or identifier
    - **Amount**: Sales amount (numeric)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit | LSTM Sales Forecasting Dashboard</div>",
    unsafe_allow_html=True
)
