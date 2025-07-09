import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

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
analysis_option = st.sidebar.selectbox(
    "Choose Analysis",
    ["Overview", "Product Sales Distribution", "Country Analysis", "Time Series Analysis", "LSTM Forecasting"]
)

# Load data from GitHub
@st.cache_data
def load_data():
    try:
        # Load main dataset
        df = pd.read_csv("https://raw.githubusercontent.com/kyashasri/sales_deploy/main/Chocolate%20Sales.csv")
        
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
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load precomputed LSTM results
@st.cache_data
def load_lstm_results():
    try:
        # Load forecast results
        forecast_df = pd.read_csv("https://raw.githubusercontent.com/kyashasri/sales_deploy/main/forecast_next_4_weeks.csv")
        
        # Load evaluation per product
        evaluation_df = pd.read_csv("https://raw.githubusercontent.com/kyashasri/sales_deploy/main/lstm_evaluation.csv")
        
        # Load overall metrics
        metrics_df = pd.read_csv("https://raw.githubusercontent.com/kyashasri/sales_deploy/main/lstm_overall_metrics.csv")
        
        return forecast_df, evaluation_df, metrics_df
    except Exception as e:
        st.error(f"Error loading LSTM results: {e}")
        return None, None, None

# Load data
df = load_data()

if df is not None:
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
        
        
        # Create pie chart
        fig_pie = px.pie(
            top_products,
            names='Product',
            values='Amount',
            hole=0.4,
            title=f'Sales Distribution: Top {top_n} Products'
        )
        
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(
            title={
                'text': f'<b>Sales Distribution: Top {top_n} Products</b>',
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
        
        # Store processed data in session state for reference
        st.session_state['interpolated_data'] = interpolated
    
    # LSTM Forecasting (Updated with precomputed results)
    elif analysis_option == "LSTM Forecasting":
        st.markdown('<div class="sub-header">🔮 LSTM Sales Forecasting</div>', unsafe_allow_html=True)
        
        # Load precomputed LSTM results
        forecast_df, evaluation_df, metrics_df = load_lstm_results()
        
        if forecast_df is not None and evaluation_df is not None and metrics_df is not None:
            st.success("✅ LSTM Forecasting Results")
            
            # Display overall metrics
            st.markdown("### Model Performance")
            
            # Extract metrics from the CSV
            mape = metrics_df['MAPE'].iloc[0]
            mae = metrics_df['MAE'].iloc[0]
            rmse = metrics_df['RMSE'].iloc[0]
            r2 = metrics_df['R2 Score'].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAPE", f"{mape:.1f}%")
            with col2:
                st.metric("MAE", f"{mae:.1f}")
            with col3:
                st.metric("RMSE", f"{rmse:.1f}","(9.5%)")
            with col4:
                st.metric("R2 Score", f"{r2:.3f}")
            
            # Actual vs Predicted comparison
            st.markdown("### Actual vs Predicted Weekly Averages")
            fig_comparison = go.Figure()
            
            product_names = evaluation_df['Product'].values
            actual_means = evaluation_df['Actual Weekly Avg'].values
            predicted_means = evaluation_df['Predicted Weekly Avg'].values
            
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
                height=600,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Display evaluation table
            st.markdown("### Model Evaluation by Product")
            st.dataframe(evaluation_df)
            
            # Forecast visualization
            st.markdown("### 4-Week Forecast Results")
            st.dataframe(forecast_df)
            
            # Interactive forecast chart
            st.markdown("### Interactive Forecast Visualization")
            
            fig_forecast = go.Figure()
            
            week_columns = [col for col in forecast_df.columns if col.startswith('Week+')]
            
            for index, row in forecast_df.iterrows():
                product = row['Product']
                values = row[week_columns].values.astype(float)
                fig_forecast.add_trace(go.Bar(
                    x=week_columns,
                    y=values,
                    name=product
                ))
            
            fig_forecast.update_layout(
                title='📦 4-Week Forecast for All Products',
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
            
            # Additional insights
            st.markdown("### Forecast Insights")
            total_forecast = forecast_df[week_columns].sum().sum()
            avg_weekly_forecast = forecast_df[week_columns].mean().mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total 4-Week Forecast", f"${total_forecast:,.2f}")
            with col2:
                st.metric("Average Weekly Forecast", f"${avg_weekly_forecast:,.2f}")
                
        else:
            st.error("Failed to load LSTM results. Please check the GitHub URLs.")

else:
    st.error("Failed to load the main dataset. Please check the GitHub URL.")

# Footer
st.markdown("---")
st.markdown("**🍫 Chocolate Sales Analysis Dashboard**")
