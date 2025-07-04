import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras with fallback
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        st.warning("⚠️ TensorFlow is not available. LSTM forecasting will be disabled.")

# Set page configuration
st.set_page_config(
    page_title="Chocolate Sales Analysis",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #D2691E;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🍫 Chocolate Sales Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
analysis_options = ["Overview", "Product Sales Distribution", "Country Analysis", "Time Series Analysis"]
if TENSORFLOW_AVAILABLE:
    analysis_options.append("LSTM Forecasting")
else:
    analysis_options.append("LSTM Forecasting (Disabled)")

analysis_option = st.sidebar.selectbox("Choose Analysis", analysis_options)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Chocolate Sales CSV", type="csv")

if uploaded_file is not None:
    # Load and process data
    @st.cache_data
    def load_and_process_data(file):
        df = pd.read_csv(file)
        
        # Data preprocessing
        df['Amount'] = df['Amount'].replace('[\$]', '', regex=True)
        df['Amount'] = df['Amount'].replace(',', '', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df.set_index('Date', inplace=True)
        
        return df
    
    df = load_and_process_data(uploaded_file)
    
    # Overview Section
    if analysis_option == "Overview":
        st.markdown('<div class="sub-header">📈 Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Revenue", f"${df['Amount'].sum():,.2f}")
        with col3:
            st.metric("Products", df['Product'].nunique())
        with col4:
            st.metric("Sales Persons", df['Sales Person'].nunique())
        
        # Display sample data
        st.markdown("### Sample Data")
        st.dataframe(df.head(10))
        
        # Basic statistics
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe())
        
        # Monthly sales distribution
        st.markdown("### Monthly Sales Distribution")
        monthly_counts = df['Month'].value_counts().sort_index()
        fig_monthly = px.bar(x=monthly_counts.index, y=monthly_counts.values, 
                            labels={'x': 'Month', 'y': 'Number of Sales'},
                            title='Sales Count by Month')
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Product Sales Distribution
    elif analysis_option == "Product Sales Distribution":
        st.markdown('<div class="sub-header">🥧 Product Sales Distribution</div>', unsafe_allow_html=True)
        
        # Aggregate total sales per product
        total_sales_per_product = df.groupby('Product')['Amount'].sum().reset_index()
        total_sales_per_product = total_sales_per_product.sort_values(by='Amount', ascending=False)
        
        # Top N products selector
        top_n = st.slider("Select number of top products to display", 3, 10, 5)
        
        top_products = total_sales_per_product.head(top_n)
        others = total_sales_per_product.iloc[top_n:]
        others_sum = others['Amount'].sum()
        
        if others_sum > 0:
            others_df = pd.DataFrame([{'Product': 'Others', 'Amount': others_sum}])
            top_products = pd.concat([top_products, others_df], ignore_index=True)
        
        # Create pie chart
        fig_pie = px.pie(
            top_products,
            names='Product',
            values='Amount',
            hole=0.4,
            title=f'Sales Distribution: Top {top_n} Products and Others'
        )
        
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(
            title={
                'text': f'<b>Sales Distribution: Top {top_n} Products and Others</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=600
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bar chart for exact values
        st.markdown("### Sales by Product (Bar Chart)")
        fig_bar = px.bar(total_sales_per_product.head(10), 
                        x='Product', y='Amount',
                        title='Top 10 Products by Sales Amount')
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Country Analysis
    elif analysis_option == "Country Analysis":
        st.markdown('<div class="sub-header">🌍 Country-wise Sales Analysis</div>', unsafe_allow_html=True)
        
        # Group by country and product
        grouped_country = df.groupby(['Country', 'Product', 'Year', 'Month'])[['Amount', 'Boxes Shipped']].sum().reset_index()
        country_product_totals = grouped_country.groupby(['Country', 'Product'])['Amount'].sum().reset_index()
        
        # Create grouped bar chart using plotly
        fig_country = px.bar(country_product_totals, 
                           x='Country', y='Amount', color='Product',
                           title='Total Sales Amount per Country for All Products',
                           barmode='group')
        fig_country.update_layout(xaxis_tickangle=-45, height=600)
        st.plotly_chart(fig_country, use_container_width=True)
        
        # Country summary table
        st.markdown("### Country Summary")
        country_summary = df.groupby('Country').agg({
            'Amount': ['sum', 'mean', 'count'],
            'Boxes Shipped': 'sum'
        }).round(2)
        country_summary.columns = ['Total Sales', 'Average Sales', 'Number of Orders', 'Total Boxes']
        st.dataframe(country_summary)
    
    # Time Series Analysis
    elif analysis_option == "Time Series Analysis":
        st.markdown('<div class="sub-header">📅 Time Series Analysis</div>', unsafe_allow_html=True)
        
        # Prepare weekly data
        @st.cache_data
        def prepare_weekly_data(df):
            grouped_date = df.groupby(['Product', df.index])[['Amount', 'Boxes Shipped']].sum().reset_index()
            grouped_date = grouped_date.sort_values(by=['Product', 'Date']).reset_index(drop=True)
            grouped_date['Date'] = pd.to_datetime(grouped_date['Date'])
            grouped_date = grouped_date.set_index('Date')
            
            weekly = (
                grouped_date.groupby('Product')
                           .resample('W-MON')[['Amount', 'Boxes Shipped']]
                           .sum()
                           .reset_index()
            )
            weekly = weekly.sort_values(by=['Product', 'Date']).reset_index(drop=True)
            return weekly
        
        weekly = prepare_weekly_data(df)
        
        # Data preprocessing for visualization
        weekly_small = weekly[['Date', 'Product', 'Amount']].copy()
        weekly_small['Amount'] = weekly_small['Amount'].replace(0, np.nan)
        
        # Outlier treatment
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
        
        # Product selection for time series
        selected_products = st.multiselect(
            "Select products to visualize:",
            options=interpolated['Product'].unique(),
            default=interpolated['Product'].unique()[:4]
        )
        
        if selected_products:
            filtered_data = interpolated[interpolated['Product'].isin(selected_products)]
            
            # Create time series plot
            fig_ts = px.line(filtered_data, x='Date', y='Amount', color='Product',
                           title='Weekly Sales Trends (Selected Products)')
            fig_ts.update_layout(height=600)
            st.plotly_chart(fig_ts, use_container_width=True)
        
        # Store processed data in session state for forecasting
        st.session_state['interpolated_data'] = interpolated
    
    # LSTM Forecasting
    elif analysis_option == "LSTM Forecasting" and TENSORFLOW_AVAILABLE:
        st.markdown('<div class="sub-header">🔮 LSTM Sales Forecasting</div>', unsafe_allow_html=True)
        
        if 'interpolated_data' not in st.session_state:
            st.warning("Please visit the 'Time Series Analysis' section first to prepare the data.")
            st.stop()
        
        interpolated = st.session_state['interpolated_data']
        
        # Forecasting parameters
        col1, col2 = st.columns(2)
        with col1:
            seq_length = st.slider("Sequence Length (weeks)", 2, 8, 4)
        with col2:
            forecast_horizon = st.slider("Forecast Horizon (weeks)", 1, 8, 4)
        
        # Function definitions
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
        
        if st.button("🚀 Run LSTM Forecasting"):
            with st.spinner("Training LSTM models and generating forecasts..."):
                # Model evaluation setup
                product_names = []
                actual_dict = {}
                predicted_dict = []
                best_epochs = []
                
                products = interpolated['Product'].unique()
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Training loop
                for idx, product in enumerate(products):
                    status_text.text(f'Training model for {product}...')
                    
                    product_df = interpolated[interpolated['Product'] == product].sort_values(by='Date')
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
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
                    
                    # LSTM Model
                    model = Sequential()
                    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                    model.add(Dropout(0.1))
                    model.add(LSTM(90))
                    model.add(Dense(1))
                    
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    
                    history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0,
                                       validation_data=(X_test, y_test), callbacks=[early_stop])
                    
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
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(products))
                
                status_text.text('Generating forecasts...')
                
                # Generate forecasts
                forecast_data = []
                for product in products:
                    product_df = interpolated[interpolated['Product'] == product].sort_values('Date')
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
                    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
                    
                    # Start forecasting recursively
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
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Product': product_names,
                    'Actual Weekly Avg': [actual_dict[p] for p in product_names],
                    'Predicted Weekly Avg': [float(p) for p in predicted_dict],
                    'Best Epoch': best_epochs
                }).round(1)
                
                # Display results
                st.success("✅ LSTM Forecasting completed!")
                
                # Model Performance
                st.markdown("### Model Performance")
                
                # Calculate metrics
                actual_means = results_df['Actual Weekly Avg'].values
                predicted_means = results_df['Predicted Weekly Avg'].values
                
                epsilon = 1e-10
                mape_avg = np.mean(np.abs((actual_means - predicted_means) / (actual_means + epsilon))) * 100
                mae_avg = mean_absolute_error(actual_means, predicted_means)
                rmse_avg = np.sqrt(mean_squared_error(actual_means, predicted_means))
                r2_avg = r2_score(actual_means, predicted_means)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAPE", f"{mape_avg:.1f}%")
                with col2:
                    st.metric("MAE", f"{mae_avg:.1f}")
                with col3:
                    st.metric("RMSE", f"{rmse_avg:.1f}")
                with col4:
                    st.metric("R² Score", f"{r2_avg:.3f}")
                
                # Actual vs Predicted comparison
                st.markdown("### Actual vs Predicted Weekly Averages")
                fig_comparison = go.Figure()
                
                x_pos = np.arange(len(product_names))
                fig_comparison.add_trace(go.Bar(
                    x=product_names,
                    y=actual_means,
                    name='Actual',
                    marker_color='lightblue'
                ))
                fig_comparison.add_trace(go.Bar(
                    x=product_names,
                    y=predicted_means,
                    name='Predicted',
                    marker_color='orange'
                ))
                
                fig_comparison.update_layout(
                    title='Actual vs Predicted Weekly Amount (All Products)',
                    xaxis_title='Product',
                    yaxis_title='Weekly Amount',
                    barmode='group',
                    height=600
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Forecast visualization
                if forecast_data:
                    week_columns = [f'Week+{i+1}' for i in range(forecast_horizon)]
                    forecast_df = pd.DataFrame(forecast_data, columns=['Product'] + week_columns)
                    
                    st.markdown("### Forecast Results")
                    st.dataframe(forecast_df)
                    
                    # Interactive forecast chart
                    st.markdown("### Interactive Forecast Visualization")
                    
                    fig_forecast = go.Figure()
                    
                    for index, row in forecast_df.iterrows():
                        product = row['Product']
                        values = row[week_columns].values.astype(float)
                        fig_forecast.add_trace(go.Bar(
                            x=week_columns,
                            y=values,
                            name=product
                        ))
                    
                    fig_forecast.update_layout(
                        title=f'📦 {forecast_horizon}-Week Forecast for All Products',
                        xaxis_title='Week',
                        yaxis_title='Forecasted Amount',
                        barmode='group',
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Download forecast
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv,
                        file_name='forecast_results.csv',
                        mime='text/csv'
                    )
                
                status_text.text('Forecasting complete!')
                progress_bar.empty()
    
    elif analysis_option == "LSTM Forecasting (Disabled)":
        st.markdown('<div class="sub-header">🔮 LSTM Sales Forecasting</div>', unsafe_allow_html=True)
        st.error("❌ LSTM Forecasting is disabled because TensorFlow is not available.")
        st.info("To enable LSTM forecasting, ensure TensorFlow is properly installed.")
        
        # Show alternative forecasting methods
        st.markdown("### Alternative Forecasting Methods")
        st.markdown("""
        While LSTM is unavailable, you can still perform forecasting using:
        - **Linear Regression** with time-based features
        - **Moving Averages** for trend analysis
        - **Seasonal Decomposition** for pattern identification
        - **ARIMA Models** (if statsmodels is available)
        """)

else:
    st.info("👆 Please upload a Chocolate Sales CSV file to begin analysis.")
    st.markdown("""
    ### Expected CSV Format:
    - **Date**: Sales date
    - **Product**: Product name
    - **Amount**: Sales amount (can include $ and commas)
    - **Country**: Country of sale
    - **Sales Person**: Salesperson name
    - **Boxes Shipped**: Number of boxes shipped
    """)

# Footer
st.markdown("---")
st.markdown("**🍫 Chocolate Sales Analysis Dashboard** | Built with Streamlit")
