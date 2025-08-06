"""
User Interface Module
Streamlit-based user interface components for the Quantum Portfolio Optimizer.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ..config.app_config import AppConfig

class AppInterface:
    """Main application interface using Streamlit"""
    
    def __init__(self):
        self.config = AppConfig()
    
    def render_header(self):
        """Render the main application header"""
        st.title(self.config.APP_TITLE)
        
        st.markdown("""
        This cutting-edge app provides portfolio optimization using both **classical** and **quantum computing** algorithms. 
        Choose your optimization method and see the advanced results!

        💡 **Quantum Advantage**: Quantum algorithms can potentially explore more solution spaces simultaneously, 
        leading to better optimization in complex portfolios with many constraints.
        """)
    
    def render_sidebar(self) -> Dict[str, Any]:
        """
        Render sidebar controls and return user inputs
        
        Returns:
            Dictionary containing all user inputs
        """
        st.sidebar.header("Portfolio Settings")
        
        # Data Source Selection
        st.sidebar.subheader("📊 Data Source")
        data_source = st.sidebar.radio(
            "Choose data source:",
            ["Real Stock Data", "Demo Data"],
            help="Real data uses Yahoo Finance, Demo data uses simulated returns"
        )
        
        user_inputs = {'data_source': data_source}
        
        if data_source == "Real Stock Data":
            user_inputs.update(self._render_stock_selection())
        else:
            user_inputs.update(self._render_demo_settings())
        
        # Optimization method selection
        user_inputs.update(self._render_optimization_settings())
        
        # Risk factor
        user_inputs['risk_factor'] = st.sidebar.slider(
            "Risk Aversion (lambda)", 0.01, 1.0, self.config.DEFAULT_RISK_FACTOR
        )
        
        # Advanced features
        user_inputs.update(self._render_advanced_features())
        
        return user_inputs
    
    def _render_stock_selection(self) -> Dict[str, Any]:
        """Render stock selection interface"""
        st.sidebar.subheader("🏢 Stock Selection")
        
        tickers_input = st.sidebar.text_area(
            "Enter stock tickers (comma-separated):",
            value=self.config.DEFAULT_TICKERS,
            help="Example: AAPL,GOOGL,MSFT,TSLA,NVDA"
        )
        
        # Date range for historical data
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=self.config.DEFAULT_LOOKBACK_DAYS))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Parse tickers
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
        
        return {
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def _render_demo_settings(self) -> Dict[str, Any]:
        """Render demo data settings"""
        st.sidebar.subheader("📈 Demo Portfolio")
        num_assets = st.sidebar.slider("Number of Assets", self.config.MIN_ASSETS, self.config.MAX_ASSETS, 4)
        
        return {'num_assets': num_assets}
    
    def _render_optimization_settings(self) -> Dict[str, Any]:
        """Render optimization method selection"""
        st.sidebar.subheader("🔬 Optimization Engine")
        
        optimization_methods = self.config.get_optimization_methods()
        if len(optimization_methods) > 2:  # Quantum available
            optimization_method = st.sidebar.selectbox(
                "Choose Optimization Method:",
                optimization_methods,
                help="Compare Both will run quantum and classical side-by-side"
            )
            
            quantum_settings = {}
            # Advanced quantum settings
            if optimization_method in ["Quantum (Qiskit)", "Compare Both"]:
                quantum_settings = self._render_quantum_settings()
            
            return {
                'optimization_method': optimization_method,
                'quantum_settings': quantum_settings
            }
        else:
            st.sidebar.info("🔧 Quantum optimization temporarily unavailable due to API updates")
            return {'optimization_method': "Classical (Scipy)", 'quantum_settings': {}}
    
    def _render_quantum_settings(self) -> Dict[str, Any]:
        """Render advanced quantum settings"""
        with st.sidebar.expander("⚙️ Advanced Quantum Settings"):
            quantum_algorithms = self.config.get_quantum_algorithms()
            algorithm = st.selectbox(
                "Quantum Algorithm:",
                quantum_algorithms,
                help="Different quantum optimization approaches"
            )
            
            if algorithm != "COBYLA (Classical-Quantum Hybrid)":
                st.info("🚧 Advanced quantum algorithms coming soon!")
            
            iterations = st.slider(
                "Optimization Iterations", 
                50, 
                self.config.QUANTUM_MAX_ITERATIONS, 
                self.config.QUANTUM_DEFAULT_ITERATIONS
            )
            show_details = st.checkbox("Show Quantum Circuit Details", False)
            
            return {
                'algorithm': algorithm,
                'iterations': iterations,
                'show_details': show_details
            }
    
    def _render_advanced_features(self) -> Dict[str, Any]:
        """Render advanced feature toggles"""
        st.sidebar.subheader("🎯 Advanced Features")
        
        return {
            'enable_risk_analysis': st.sidebar.checkbox("📊 Enhanced Risk Analysis", True),
            'enable_backtesting': st.sidebar.checkbox("⏱️ Historical Backtesting", False),
            'enable_monte_carlo': st.sidebar.checkbox("🎲 Monte Carlo Simulation", False)
        }
    
    def display_data_overview(self, data_result: Dict[str, Any]):
        """Display portfolio data overview"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Expected Returns")
            if data_result['data_source'] == 'real':
                returns_df = pd.DataFrame({
                    'Stock': data_result['asset_names'], 
                    'Expected Return (Annual)': data_result['expected_returns']
                })
            else:
                returns_df = pd.DataFrame({
                    'Asset': data_result['asset_names'], 
                    'Expected Return': data_result['expected_returns']
                })
            st.dataframe(returns_df, use_container_width=True)
        
        with col2:
            st.subheader("Risk (Standard Deviation)")
            risk_values = np.sqrt(np.diag(data_result['covariance_matrix']))
            if data_result['data_source'] == 'real':
                risk_df = pd.DataFrame({
                    'Stock': data_result['asset_names'], 
                    'Risk (σ) Annual': risk_values
                })
            else:
                risk_df = pd.DataFrame({
                    'Asset': data_result['asset_names'], 
                    'Risk (σ)': risk_values
                })
            st.dataframe(risk_df, use_container_width=True)
    
    def display_price_charts(self, price_data: pd.DataFrame, asset_names: List[str]):
        """Display price performance charts for real stock data"""
        st.subheader("📈 Recent Price Performance")
        
        # Normalize prices to start at 100 for comparison
        normalized_prices = (price_data / price_data.iloc[0]) * 100
        
        fig = go.Figure()
        for ticker in asset_names:
            if ticker in normalized_prices.columns:
                fig.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[ticker],
                    mode='lines',
                    name=ticker
                ))
        
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Price (Normalized)",
            height=self.config.CHART_HEIGHT
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_optimization_results(self, result: Dict[str, Any]):
        """Display optimization results and comparison"""
        if result['method_used'] == "Compare Both":
            self._display_comparison_results(result)
        else:
            self._display_single_optimization_results(result)
    
    def _display_comparison_results(self, result: Dict[str, Any]):
        """Display comparison results for quantum vs classical"""
        st.success("🌟 Comparative Optimization Complete!")
        
        # Comparison metrics
        st.subheader("⚖️ Quantum vs Classical Comparison")
        comparison = result['comparison_results']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Return", 
                     f"Classical: {comparison['classical']['portfolio_return']:.2%}",
                     f"Quantum: {comparison['quantum']['portfolio_return']:.2%}")
        with col2:
            st.metric("Portfolio Risk", 
                     f"Classical: {comparison['classical']['portfolio_risk']:.2%}",
                     f"Quantum: {comparison['quantum']['portfolio_risk']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", 
                     f"Classical: {comparison['classical_sharpe']:.3f}",
                     f"Quantum: {comparison['quantum_sharpe']:.3f}")
        
        # Winner announcement
        if comparison['winner'] == 'quantum':
            st.success("🏆 **Quantum Advantage**: Quantum optimization achieved a higher Sharpe ratio!")
        else:
            st.info("📊 **Classical Performance**: Classical optimization performed better in this case.")
    
    def _display_single_optimization_results(self, result: Dict[str, Any]):
        """Display results for single optimization method"""
        if result['used_quantum']:
            st.success("🌟 Quantum Optimization Complete! Advanced quantum algorithms explored the solution space.")
            method_text = "**🔬 Method Used:** Quantum Computing (Qiskit) - *Next-generation optimization*"
        else:
            st.success("✅ Classical Optimization Complete! Proven mathematical optimization applied.")
            method_text = "**📊 Method Used:** Classical Computing (SciPy) - *Proven mathematical optimization*"
        
        # Method indicator
        col_method, col_spacer = st.columns([3, 1])
        with col_method:
            st.markdown(method_text)
        
        # Results display
        sharpe_ratio = result['portfolio_return'] / result['portfolio_risk'] if result['portfolio_risk'] > 0 else 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Return", f"{result['portfolio_return']:.2%}")
        with col2:
            st.metric("Portfolio Risk", f"{result['portfolio_risk']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
        
        # Show quantum details if available
        if result.get('quantum_details') and result['used_quantum']:
            self._display_quantum_details(result['quantum_details'])
    
    def _display_quantum_details(self, quantum_details: Dict[str, Any]):
        """Display quantum optimization details"""
        with st.expander("🔬 Quantum Optimization Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Problem Size", quantum_details.get('problem_size', 'N/A'))
                st.metric("Variables", quantum_details.get('variables', 'N/A'))
            with col2:
                st.metric("Constraints", quantum_details.get('constraints', 'N/A'))
                st.metric("Quadratic Terms", quantum_details.get('quadratic_terms', 'N/A'))
    
    def display_portfolio_allocation(self, result: Dict[str, Any], asset_names: List[str]):
        """Display portfolio allocation chart and table"""
        st.subheader("🎯 Optimal Portfolio Allocation")
        
        weights = result['weights']
        allocation_df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights,
            'Weight (%)': weights * 100
        })
        
        # Filter out very small allocations for cleaner display
        allocation_df = allocation_df[allocation_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(allocation_df, use_container_width=True)
        
        with col2:
            # Pie chart of allocation
            fig = px.pie(allocation_df, values='Weight', names='Asset', 
                        title="Portfolio Allocation")
            if result['used_quantum']:
                fig.update_layout(title="🌟 Quantum-Optimized Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_return_chart(self, result: Dict[str, Any], data_result: Dict[str, Any]):
        """Display risk-return scatter plot"""
        st.subheader("📈 Risk-Return Analysis")
        
        mu = data_result['expected_returns']
        cov = data_result['covariance_matrix']
        asset_names = data_result['asset_names']
        weights = result['weights']
        
        fig = go.Figure()
        
        # Individual assets
        fig.add_trace(go.Scatter(
            x=np.sqrt(np.diag(cov)),
            y=mu,
            mode='markers+text',
            text=asset_names,
            textposition="top center",
            name='Individual Assets',
            marker=dict(size=10, color='lightblue')
        ))
        
        # Optimized portfolio
        portfolio_color = '#FFD700' if result['used_quantum'] else '#FF4B4B'
        portfolio_symbol = 'star' if result['used_quantum'] else 'diamond'
        
        fig.add_trace(go.Scatter(
            x=[result['portfolio_risk']],
            y=[result['portfolio_return']],
            mode='markers+text',
            text=['Optimized Portfolio'],
            textposition="top center",
            name='Optimized Portfolio',
            marker=dict(size=15, color=portfolio_color, symbol=portfolio_symbol)
        ))
        
        fig.update_layout(
            xaxis_title='Risk (Standard Deviation)',
            yaxis_title='Expected Return',
            title=f'Risk-Return Profile ({result["method_used"]})',
            height=self.config.CHART_HEIGHT
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_analysis(self, risk_analysis: Dict[str, Any]):
        """Display enhanced risk analysis results"""
        st.subheader("🔍 Enhanced Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("VaR (95%)", f"{risk_analysis['var_95']:.2%}", 
                     help="Value at Risk at 95% confidence")
            st.metric("CVaR (95%)", f"{risk_analysis['cvar_95']:.2%}", 
                     help="Conditional Value at Risk at 95%")
        with col2:
            st.metric("VaR (99%)", f"{risk_analysis['var_99']:.2%}", 
                     help="Value at Risk at 99% confidence")
            st.metric("CVaR (99%)", f"{risk_analysis['cvar_99']:.2%}", 
                     help="Conditional Value at Risk at 99%")
        with col3:
            st.metric("Max Drawdown", f"{risk_analysis['max_drawdown']:.2%}", 
                     help="Maximum historical drawdown")
        
        # Correlation heatmap
        with st.expander("📊 Asset Correlation Analysis"):
            if len(risk_analysis['asset_names']) > 1:
                fig_heatmap = px.imshow(
                    risk_analysis['correlation_matrix'],
                    labels=dict(x="Assets", y="Assets", color="Correlation"),
                    x=risk_analysis['asset_names'],
                    y=risk_analysis['asset_names'],
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig_heatmap.update_layout(title="Asset Correlation Matrix")
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def display_monte_carlo_results(self, mc_results: Dict[str, Any]):
        """Display Monte Carlo simulation results"""
        st.subheader("🎲 Monte Carlo Portfolio Simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            # Histogram of simulated returns
            fig_hist = px.histogram(
                mc_results['simulations'], 
                nbins=50, 
                title="Distribution of Simulated Annual Returns",
                labels={'value': 'Annual Return', 'count': 'Frequency'}
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.metric("Expected Return", f"{mc_results['expected_return']:.2%}")
            st.metric("Standard Deviation", f"{mc_results['std_deviation']:.2%}")
            st.metric("5th Percentile", f"{mc_results['percentile_5']:.2%}")
            st.metric("95th Percentile", f"{mc_results['percentile_95']:.2%}")
            st.metric("Probability of Loss", f"{mc_results['probability_of_loss']:.1%}")
    
    def display_ai_insights(self, insights: Dict[str, Any]):
        """Display AI-powered portfolio insights"""
        st.subheader("🤖 AI-Powered Portfolio Insights")
        
        # Portfolio composition analysis
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Diversification Score", f"{insights['diversification_score']:.3f}", 
                     help="Higher scores indicate better diversification (max = 1)")
            st.metric("Top Risk Contributor", insights['top_risk_contributor'], 
                     f"{insights['risk_contributions'].max():.1%} of portfolio risk")
        
        with col2:
            st.metric("Top Return Contributor", insights['top_return_contributor'], 
                     f"{insights['return_contributions'].max():.2%} expected return")
            st.metric("Efficiency vs Equal Weight", f"{insights['efficiency_ratio']:.2f}", 
                     help="How much better this portfolio is vs equal weighting")
        
        # Smart recommendations
        with st.expander("💡 AI Recommendations"):
            for recommendation in insights['recommendations']:
                st.write(recommendation)
    
    def display_quantum_messaging(self, result: Dict[str, Any], user_inputs: Dict[str, Any]):
        """Display quantum advantage messaging"""
        if result['used_quantum']:
            st.info("🌟 **Quantum Advantage**: This optimization utilized quantum computing principles to explore multiple solution paths simultaneously, potentially finding superior allocations.")
        elif self.config.is_quantum_available() and user_inputs['optimization_method'] == "Classical (Scipy)":
            st.info("💡 **Try Quantum**: Switch to quantum optimization to leverage advanced quantum algorithms for potentially better results!")
    
    def display_getting_started(self):
        """Display getting started information"""
        st.info("📊 Set your parameters and click 'Optimize Portfolio' to begin.")
        
        # Display quantum advantage information
        with st.expander("🔬 Why Choose Quantum Optimization?"):
            st.markdown("""
            **Quantum Computing Advantages in Portfolio Optimization:**
            
            - **Parallel Processing**: Quantum algorithms can explore multiple portfolio combinations simultaneously
            - **Complex Constraint Handling**: Better at managing multiple investment constraints and correlations
            - **Future-Ready**: Positions your investment strategy with cutting-edge technology
            - **Competitive Edge**: Quantum optimization may identify opportunities that classical methods miss
            
            *As quantum hardware improves, these advantages will become even more pronounced.*
            """)
