"""
Quantum Portfolio Optimization - Main Application
A cutting-edge portfolio optimization platform using quantum computing.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.ui.app_interface import AppInterface
from src.data.data_manager import DataManager
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from src.analysis.risk_analyzer import RiskAnalyzer
from src.analysis.ai_insights import AIInsights
from src.config.app_config import AppConfig

def main():
    """Main application entry point"""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Quantum Portfolio Optimizer", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize application components
    config = AppConfig()
    ui = AppInterface()
    data_manager = DataManager()
    optimizer = PortfolioOptimizer()
    risk_analyzer = RiskAnalyzer()
    ai_insights = AIInsights()
    
    # Render main UI
    ui.render_header()
    
    # Get user inputs from sidebar
    user_inputs = ui.render_sidebar()
    
    # Load and display data
    data_result = data_manager.get_portfolio_data(
        data_source=user_inputs['data_source'],
        tickers=user_inputs.get('tickers', []),
        start_date=user_inputs.get('start_date'),
        end_date=user_inputs.get('end_date'),
        num_assets=user_inputs.get('num_assets', 4)
    )
    
    if data_result:
        # Display data overview
        ui.display_data_overview(data_result)
        
        # Display price charts if real data
        if data_result.get('price_data') is not None:
            ui.display_price_charts(data_result['price_data'], data_result['asset_names'])
        
        # Portfolio optimization
        if st.button("🎯 Optimize Portfolio", type="primary"):
            with st.spinner(f"Running {user_inputs['optimization_method']} optimization..."):
                
                # Run optimization
                optimization_result = optimizer.optimize_portfolio(
                    mu=data_result['expected_returns'],
                    cov=data_result['covariance_matrix'],
                    method=user_inputs['optimization_method'],
                    risk_factor=user_inputs['risk_factor'],
                    quantum_settings=user_inputs.get('quantum_settings', {})
                )
                
                # Display results
                ui.display_optimization_results(optimization_result)
                ui.display_portfolio_allocation(optimization_result, data_result['asset_names'])
                ui.display_risk_return_chart(optimization_result, data_result)
                
                # Enhanced analysis (if enabled)
                if user_inputs.get('enable_risk_analysis', False):
                    risk_analysis = risk_analyzer.analyze_portfolio(
                        weights=optimization_result['weights'],
                        returns_data=data_result.get('returns_data'),
                        asset_names=data_result['asset_names']
                    )
                    ui.display_risk_analysis(risk_analysis)
                
                # Monte Carlo simulation (if enabled)
                if user_inputs.get('enable_monte_carlo', False):
                    mc_results = risk_analyzer.monte_carlo_simulation(
                        mu=data_result['expected_returns'],
                        cov=data_result['covariance_matrix'],
                        weights=optimization_result['weights']
                    )
                    ui.display_monte_carlo_results(mc_results)
                
                # AI-powered insights
                insights = ai_insights.generate_insights(
                    optimization_result=optimization_result,
                    data_result=data_result,
                    user_inputs=user_inputs
                )
                ui.display_ai_insights(insights)
                
                # Quantum advantage messaging
                ui.display_quantum_messaging(optimization_result, user_inputs)
    
    else:
        # Show getting started info
        ui.display_getting_started()

if __name__ == "__main__":
    main()
