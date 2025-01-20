import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from market_sentinel import MarketSentinel
from portfolio_architect import PortfolioArchitect
from visualizer import MarketVisualizer

def main():
    st.set_page_config(page_title="Market Sentinel", layout="wide")
    
    # Header
    st.title("🎯 Market Sentinel")
    st.markdown("### ML-Powered Market Analysis & Portfolio Optimization")
    
    # Initialize components
    sentinel = MarketSentinel()
    architect = PortfolioArchitect()
    visualizer = MarketVisualizer(theme='light')
    
    # Sidebar settings
    st.sidebar.header("Settings")
    lookback_years = st.sidebar.slider("Years of Data", 2, 10, 5)
    
    try:
        # Load data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*lookback_years)
        
        with st.spinner("Fetching market data..."):
            market_data = sentinel.fetch_market_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        # Train model
        with st.spinner("Training ML model..."):
            features, labels = sentinel.prepare_features(market_data)
            metrics = sentinel.train_model(features, labels)
            
            # Display model metrics
            st.header("🤖 ML Model Performance")
            st.code(metrics['classification_report'])
            
            # Display feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': metrics['feature_importance'].keys(),
                'Importance': metrics['feature_importance'].values()
            }).sort_values('Importance', ascending=False)
            st.bar_chart(importance_df.set_index('Feature'))
        
        # Calculate features and portfolio
        with st.spinner("Analyzing market conditions..."):
            features = sentinel.calculate_market_features(market_data)
            market_analysis = sentinel.analyze_market_conditions(market_data)
            portfolio = architect.optimize_portfolio(market_data, features)
        
        # Display analysis
        st.header("📈 Market Analysis")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Crash Probability",
                f"{market_analysis['crash_probability']:.1%}",
                delta=f"{market_analysis['crash_probability'] - 0.5:.1%}"
            )
        with col2:
            st.metric("Market Regime", market_analysis['market_regime'].title())
        with col3:
            st.metric(
                "Stress Level",
                f"{market_analysis['stress_level']:.2f}"
            )
        
        # Visualizations
        market_fig = visualizer.plot_market_dashboard(market_data, features)
        st.plotly_chart(market_fig, use_container_width=True)
        
        # Portfolio analysis
        st.header("💼 Portfolio Analysis")
        portfolio_fig = visualizer.plot_portfolio_analysis(portfolio, market_data)
        st.plotly_chart(portfolio_fig, use_container_width=True)
        
        # Insights and recommendations
        st.header("📋 Insights & Recommendations")
        
        # Display ML-based insights
        st.subheader("Market Insights")
        for insight in market_analysis['insights']:
            st.info(insight)
        
        # Display portfolio recommendations
        st.subheader("Portfolio Recommendations")
        for step in portfolio['implementation']:
            st.success(step)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again with a different date range.")

if __name__ == "__main__":
    main()