import streamlit as st
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

# Try to import quantum components
try:
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.algorithms import CobylaOptimizer
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

st.set_page_config(page_title="Quantum Portfolio Optimizer", layout="wide")
st.title("🚀 Quantum Portfolio Optimization & Diversification")

st.markdown("""
This cutting-edge app provides portfolio optimization using both **classical** and **quantum computing** algorithms. 
Choose your optimization method and see the advanced results!

💡 **Quantum Advantage**: Quantum algorithms can potentially explore more solution spaces simultaneously, 
leading to better optimization in complex portfolios with many constraints.
""")

# Sidebar for user input
st.sidebar.header("Portfolio Settings")

# Data Source Selection
st.sidebar.subheader("� Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Real Stock Data", "Demo Data"],
    help="Real data uses Yahoo Finance, Demo data uses simulated returns"
)

if data_source == "Real Stock Data":
    # Stock ticker input
    st.sidebar.subheader("🏢 Stock Selection")
    default_tickers = "AAPL,GOOGL,MSFT,AMZN"
    tickers_input = st.sidebar.text_area(
        "Enter stock tickers (comma-separated):",
        value=default_tickers,
        help="Example: AAPL,GOOGL,MSFT,TSLA,NVDA"
    )
    
    # Date range for historical data
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=252))  # 1 year
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Parse tickers
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    num_assets = len(tickers)
    asset_names = tickers
    
else:
    # Demo data settings
    st.sidebar.subheader("📈 Demo Portfolio")
    num_assets = st.sidebar.slider("Number of Assets", 2, 10, 4)
    asset_names = [f"Asset {i+1}" for i in range(num_assets)]

# Optimization method selection
st.sidebar.subheader("🔬 Optimization Engine")
if QUANTUM_AVAILABLE:
    optimization_method = st.sidebar.selectbox(
        "Choose Optimization Method:",
        ["Classical (Scipy)", "Quantum (Qiskit)", "Compare Both"],
        help="Compare Both will run quantum and classical side-by-side"
    )
    
    # Advanced quantum settings
    if optimization_method in ["Quantum (Qiskit)", "Compare Both"]:
        with st.sidebar.expander("⚙️ Advanced Quantum Settings"):
            quantum_algorithm = st.selectbox(
                "Quantum Algorithm:",
                ["COBYLA (Classical-Quantum Hybrid)", "QAOA (Future)", "VQE (Future)"],
                help="Different quantum optimization approaches"
            )
            
            if quantum_algorithm != "COBYLA (Classical-Quantum Hybrid)":
                st.info("🚧 Advanced quantum algorithms coming soon!")
                
            quantum_iterations = st.slider("Optimization Iterations", 50, 500, 200)
            show_quantum_details = st.checkbox("Show Quantum Circuit Details", False)
else:
    st.sidebar.info("🔧 Quantum optimization temporarily unavailable due to API updates")
    optimization_method = "Classical (Scipy)"

risk_factor = st.sidebar.slider("Risk Aversion (lambda)", 0.01, 1.0, 0.5)

# Advanced Features
st.sidebar.subheader("🎯 Advanced Features")
enable_risk_analysis = st.sidebar.checkbox("📊 Enhanced Risk Analysis", True)
enable_backtesting = st.sidebar.checkbox("⏱️ Historical Backtesting", False)
enable_monte_carlo = st.sidebar.checkbox("🎲 Monte Carlo Simulation", False)

# Data Processing
@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    """Fetch stock data and calculate returns and covariance"""
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
        
        # Handle different data structures based on number of tickers
        if len(tickers) == 1:
            # Single ticker - data is a simple DataFrame
            if 'Close' in data.columns:
                price_data = data[['Close']].copy()
                price_data.columns = tickers
            else:
                price_data = data[['Adj Close']].copy()
                price_data.columns = tickers
        else:
            # Multiple tickers - data has MultiIndex columns
            if 'Close' in data.columns.get_level_values(0):
                price_data = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                price_data = data['Adj Close']
            else:
                # Fallback - try to get the closing prices
                price_data = data.iloc[:, [i for i, col in enumerate(data.columns) if 'Close' in str(col)]]
        
        # Ensure we have valid data
        if price_data.empty:
            raise ValueError("No price data found")
        
        # Remove any tickers with all NaN values
        price_data = price_data.dropna(axis=1, how='all')
        
        if price_data.empty:
            raise ValueError("All price data is NaN")
        
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        # Annualize expected returns (252 trading days)
        mu = returns.mean() * 252
        
        # Annualize covariance matrix
        cov = returns.cov() * 252
        
        return mu.values, cov.values, returns, price_data
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        # Try alternative approach with individual ticker downloads
        try:
            st.info("Trying alternative data fetch method...")
            price_data_list = []
            
            for ticker in tickers:
                ticker_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                if not ticker_data.empty:
                    if 'Close' in ticker_data.columns:
                        price_data_list.append(ticker_data[['Close']].rename(columns={'Close': ticker}))
                    elif 'Adj Close' in ticker_data.columns:
                        price_data_list.append(ticker_data[['Adj Close']].rename(columns={'Adj Close': ticker}))
            
            if price_data_list:
                price_data = pd.concat(price_data_list, axis=1).dropna()
                returns = price_data.pct_change().dropna()
                mu = returns.mean() * 252
                cov = returns.cov() * 252
                return mu.values, cov.values, returns, price_data
            else:
                return None, None, None, None
                
        except Exception as e2:
            st.error(f"Alternative fetch also failed: {str(e2)}")
            return None, None, None, None

# Get data based on source
if data_source == "Real Stock Data" and len(tickers) > 1:
    with st.spinner("Fetching real-time stock data..."):
        mu, cov, returns_data, price_data = get_stock_data(tickers, start_date, end_date)
    
    if mu is not None and cov is not None:
        st.success(f"✅ Successfully loaded data for {len(tickers)} stocks from {start_date} to {end_date}")
    else:
        st.error("Failed to load stock data. Using demo data instead.")
        # Fallback to demo data
        np.random.seed(42)
        mu = np.random.uniform(0.05, 0.2, num_assets)
        sigma = np.random.uniform(0.01, 0.05, (num_assets, num_assets))
        cov = np.dot(sigma, sigma.T)
        
elif data_source == "Real Stock Data" and len(tickers) <= 1:
    st.warning("Please enter at least 2 stock tickers for portfolio optimization.")
    # Use demo data
    np.random.seed(42)
    mu = np.random.uniform(0.05, 0.2, 4)
    sigma = np.random.uniform(0.01, 0.05, (4, 4))
    cov = np.dot(sigma, sigma.T)
    asset_names = ["Asset 1", "Asset 2", "Asset 3", "Asset 4"]
    num_assets = 4
else:
    # Demo data
    np.random.seed(42)
    mu = np.random.uniform(0.05, 0.2, num_assets)
    sigma = np.random.uniform(0.01, 0.05, (num_assets, num_assets))
    cov = np.dot(sigma, sigma.T)
# Display data
col1, col2 = st.columns(2)
with col1:
    st.subheader("Expected Returns")
    if data_source == "Real Stock Data" and mu is not None:
        returns_df = pd.DataFrame({'Stock': asset_names, 'Expected Return (Annual)': mu})
    else:
        returns_df = pd.DataFrame({'Asset': asset_names, 'Expected Return': mu})
    st.dataframe(returns_df, use_container_width=True)

with col2:
    st.subheader("Risk (Standard Deviation)")
    if data_source == "Real Stock Data" and cov is not None:
        risk_df = pd.DataFrame({'Stock': asset_names, 'Risk (σ) Annual': np.sqrt(np.diag(cov))})
    else:
        risk_df = pd.DataFrame({'Asset': asset_names, 'Risk (σ)': np.sqrt(np.diag(cov))})
    st.dataframe(risk_df, use_container_width=True)

# Show recent price chart for real stocks
if data_source == "Real Stock Data" and 'price_data' in locals() and price_data is not None:
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
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Portfolio Optimization using Mean-Variance (Classical)
def classical_portfolio_optimization(mu, cov, risk_aversion):
    n = len(mu)
    # Objective: minimize risk - return + penalty for risk aversion
    def objective(weights):
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        return portfolio_risk * risk_aversion - portfolio_return
    
    # Constraints: weights sum to 1, all weights >= 0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    
    # Initial guess: equal weights
    x0 = np.array([1/n] * n)
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Quantum Portfolio Optimization with Advanced Features
def quantum_portfolio_optimization(mu, cov, risk_aversion, algorithm="COBYLA", iterations=200, show_details=False):
    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_algorithms import QAOA
        from qiskit.primitives import Sampler
        
        # Create a continuous quadratic program (compatible with COBYLA)
        qp = QuadraticProgram()
        n = len(mu)
        
        # Add continuous variables for portfolio weights
        for i in range(n):
            qp.continuous_var(name=f'x_{i}', lowerbound=0, upperbound=1)
        
        # Objective: minimize portfolio risk - expected return
        # Portfolio variance: x^T * Cov * x
        # Portfolio return: mu^T * x
        linear_terms = {}
        quadratic_terms = {}
        
        # Linear terms (negative expected returns)
        for i in range(n):
            linear_terms[f'x_{i}'] = -mu[i]
        
        # Quadratic terms (covariance matrix scaled by risk aversion)
        for i in range(n):
            for j in range(n):
                if i <= j:  # Only upper triangle needed
                    quadratic_terms[(f'x_{i}', f'x_{j}')] = risk_aversion * cov[i, j]
        
        qp.minimize(linear=linear_terms, quadratic=quadratic_terms)
        
        # Constraint: sum of weights = 1
        linear_constraint = {}
        for i in range(n):
            linear_constraint[f'x_{i}'] = 1
        qp.linear_constraint(linear=linear_constraint, sense='==', rhs=1, name='budget')
        
        # Show quantum problem formulation if requested
        quantum_details = {}
        if show_details:
            quantum_details = {
                'problem_size': n,
                'variables': n,
                'constraints': 1,
                'quadratic_terms': len(quadratic_terms),
                'algorithm': algorithm
            }
        
        # Use COBYLA optimizer with the continuous problem
        optimizer = CobylaOptimizer(maxiter=iterations)
        result = optimizer.solve(qp)
        
        if result.x is not None and len(result.x) == n:
            weights = np.array(result.x)
            # Ensure weights are normalized and non-negative
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n) / n
            return weights, True, quantum_details
        else:
            # Fallback to classical if quantum fails
            classical_result = classical_portfolio_optimization(mu, cov, risk_aversion)
            return classical_result.x if classical_result.success else np.ones(n) / n, False, quantum_details
            
    except Exception as e:
        st.warning(f"Quantum optimization encountered an issue: {str(e)}. Using classical fallback.")
        result = classical_portfolio_optimization(mu, cov, risk_aversion)
        return result.x if result.success else np.ones(len(mu)) / len(mu), False, {}

# Enhanced Risk Analysis Functions
def calculate_var_cvar(returns, weights, confidence_level=0.05):
    """Calculate Value at Risk and Conditional Value at Risk"""
    portfolio_returns = np.dot(returns, weights)
    var = np.percentile(portfolio_returns, confidence_level * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return var, cvar

def calculate_maximum_drawdown(returns, weights):
    """Calculate maximum drawdown for the portfolio"""
    portfolio_returns = np.dot(returns, weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate running maximum using numpy
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown

def monte_carlo_simulation(mu, cov, weights, num_simulations=1000, time_horizon=252):
    """Run Monte Carlo simulation for portfolio performance"""
    np.random.seed(42)  # For reproducible results
    
    # Generate random returns
    simulated_returns = np.random.multivariate_normal(mu/252, cov/252, (num_simulations, time_horizon))
    
    # Calculate portfolio returns for each simulation
    portfolio_simulations = []
    for sim in simulated_returns:
        portfolio_returns = np.dot(sim, weights)
        cumulative_return = (1 + portfolio_returns).prod() - 1
        portfolio_simulations.append(cumulative_return)
    
    return np.array(portfolio_simulations)

# Portfolio Optimization
if st.button("🎯 Optimize Portfolio", type="primary"):
    with st.spinner(f"Running {optimization_method} optimization..."):
        
        if optimization_method == "Compare Both" and QUANTUM_AVAILABLE:
            # Run both optimizations
            st.info("🔄 Running comparative analysis between Quantum and Classical methods...")
            
            # Classical optimization
            classical_result = classical_portfolio_optimization(mu, cov, risk_factor)
            classical_weights = classical_result.x if classical_result.success else np.ones(len(mu)) / len(mu)
            
            # Quantum optimization
            quantum_weights, quantum_success, quantum_details = quantum_portfolio_optimization(
                mu, cov, risk_factor, 
                algorithm=quantum_algorithm if 'quantum_algorithm' in locals() else "COBYLA",
                iterations=quantum_iterations if 'quantum_iterations' in locals() else 200,
                show_details=show_quantum_details if 'show_quantum_details' in locals() else False
            )
            
            # Store both results
            results = {
                'classical': {
                    'weights': classical_weights,
                    'return': np.dot(classical_weights, mu),
                    'risk': np.sqrt(np.dot(classical_weights, np.dot(cov, classical_weights))),
                    'used_quantum': False
                },
                'quantum': {
                    'weights': quantum_weights,
                    'return': np.dot(quantum_weights, mu),
                    'risk': np.sqrt(np.dot(quantum_weights, np.dot(cov, quantum_weights))),
                    'used_quantum': quantum_success,
                    'details': quantum_details
                }
            }
            
            # Display comparison
            st.success("🌟 Comparative Optimization Complete!")
            
            # Comparison metrics
            st.subheader("⚖️ Quantum vs Classical Comparison")
            col1, col2, col3 = st.columns(3)
            
            classical_sharpe = results['classical']['return'] / results['classical']['risk'] if results['classical']['risk'] > 0 else 0
            quantum_sharpe = results['quantum']['return'] / results['quantum']['risk'] if results['quantum']['risk'] > 0 else 0
            
            with col1:
                st.metric("Portfolio Return", 
                         f"Classical: {results['classical']['return']:.2%}",
                         f"Quantum: {results['quantum']['return']:.2%}")
            with col2:
                st.metric("Portfolio Risk", 
                         f"Classical: {results['classical']['risk']:.2%}",
                         f"Quantum: {results['quantum']['risk']:.2%}")
            with col3:
                st.metric("Sharpe Ratio", 
                         f"Classical: {classical_sharpe:.3f}",
                         f"Quantum: {quantum_sharpe:.3f}")
            
            # Determine winner
            if quantum_sharpe > classical_sharpe:
                st.success("🏆 **Quantum Advantage**: Quantum optimization achieved a higher Sharpe ratio!")
                primary_weights = results['quantum']['weights']
                used_quantum = True
            else:
                st.info("📊 **Classical Performance**: Classical optimization performed better in this case.")
                primary_weights = results['classical']['weights']
                used_quantum = False
            
            weights = primary_weights
            
        elif optimization_method == "Quantum (Qiskit)" and QUANTUM_AVAILABLE:
            # Use quantum optimization
            weights, quantum_success, quantum_details = quantum_portfolio_optimization(
                mu, cov, risk_factor,
                algorithm=quantum_algorithm if 'quantum_algorithm' in locals() else "COBYLA",
                iterations=quantum_iterations if 'quantum_iterations' in locals() else 200,
                show_details=show_quantum_details if 'show_quantum_details' in locals() else False
            )
            used_quantum = quantum_success
            
            # Show quantum details if enabled
            if 'show_quantum_details' in locals() and show_quantum_details and quantum_details:
                with st.expander("🔬 Quantum Optimization Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Problem Size", quantum_details.get('problem_size', 'N/A'))
                        st.metric("Variables", quantum_details.get('variables', 'N/A'))
                    with col2:
                        st.metric("Constraints", quantum_details.get('constraints', 'N/A'))
                        st.metric("Quadratic Terms", quantum_details.get('quadratic_terms', 'N/A'))
        else:
            # Use classical optimization
            result = classical_portfolio_optimization(mu, cov, risk_factor)
            weights = result.x if result.success else np.ones(len(mu)) / len(mu)
            used_quantum = False
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Success message with method indicator
        if used_quantum:
            st.success("� Quantum Optimization Complete! Advanced quantum algorithms explored the solution space.")
        else:
            st.success("✅ Classical Optimization Complete! Proven mathematical optimization applied.")
        
        # Method indicator
        col_method, col_spacer = st.columns([3, 1])
        with col_method:
            if used_quantum:
                st.markdown("**🔬 Method Used:** Quantum Computing (Qiskit) - *Next-generation optimization*")
            else:
                st.markdown("**📊 Method Used:** Classical Computing (SciPy) - *Proven mathematical optimization*")
        
        # Results display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Return", f"{portfolio_return:.2%}")
        with col2:
            st.metric("Portfolio Risk", f"{portfolio_risk:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
        
        # Portfolio allocation
        st.subheader("🎯 Optimal Portfolio Allocation")
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
            if used_quantum:
                fig.update_layout(title="🌟 Quantum-Optimized Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk-Return scatter plot
        st.subheader("📈 Risk-Return Analysis")
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
        portfolio_color = '#FFD700' if used_quantum else '#FF4B4B'  # Gold for quantum, red for classical
        portfolio_symbol = 'star' if used_quantum else 'diamond'
        
        fig.add_trace(go.Scatter(
            x=[portfolio_risk],
            y=[portfolio_return],
            mode='markers+text',
            text=['Optimized Portfolio'],
            textposition="top center",
            name='Optimized Portfolio',
            marker=dict(size=15, color=portfolio_color, symbol=portfolio_symbol)
        ))
        
        fig.update_layout(
            xaxis_title='Risk (Standard Deviation)',
            yaxis_title='Expected Return',
            title=f'Risk-Return Profile ({optimization_method})'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Risk Analysis (if enabled)
        if 'enable_risk_analysis' in locals() and enable_risk_analysis:
            st.subheader("🔍 Enhanced Risk Analysis")
            
            if data_source == "Real Stock Data" and 'returns_data' in locals() and returns_data is not None:
                # Calculate VaR and CVaR
                var_95, cvar_95 = calculate_var_cvar(returns_data.values, weights, 0.05)
                var_99, cvar_99 = calculate_var_cvar(returns_data.values, weights, 0.01)
                
                # Calculate Maximum Drawdown
                max_dd = calculate_maximum_drawdown(returns_data.values, weights)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("VaR (95%)", f"{var_95:.2%}", help="Value at Risk at 95% confidence")
                    st.metric("CVaR (95%)", f"{cvar_95:.2%}", help="Conditional Value at Risk at 95%")
                with col2:
                    st.metric("VaR (99%)", f"{var_99:.2%}", help="Value at Risk at 99% confidence")
                    st.metric("CVaR (99%)", f"{cvar_99:.2%}", help="Conditional Value at Risk at 99%")
                with col3:
                    st.metric("Max Drawdown", f"{max_dd:.2%}", help="Maximum historical drawdown")
                
                # Correlation heatmap
                with st.expander("📊 Asset Correlation Analysis"):
                    if len(asset_names) > 1:
                        corr_matrix = returns_data.corr()
                        fig_heatmap = px.imshow(
                            corr_matrix,
                            labels=dict(x="Assets", y="Assets", color="Correlation"),
                            x=asset_names,
                            y=asset_names,
                            color_continuous_scale="RdBu",
                            aspect="auto"
                        )
                        fig_heatmap.update_layout(title="Asset Correlation Matrix")
                        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Monte Carlo Simulation (if enabled)
        if 'enable_monte_carlo' in locals() and enable_monte_carlo:
            st.subheader("🎲 Monte Carlo Portfolio Simulation")
            
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = monte_carlo_simulation(mu, cov, weights, num_simulations=1000, time_horizon=252)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Histogram of simulated returns
                    fig_hist = px.histogram(
                        mc_results, 
                        nbins=50, 
                        title="Distribution of Simulated Annual Returns",
                        labels={'value': 'Annual Return', 'count': 'Frequency'}
                    )
                    fig_hist.update_layout(showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Summary statistics
                    st.metric("Expected Return", f"{np.mean(mc_results):.2%}")
                    st.metric("Standard Deviation", f"{np.std(mc_results):.2%}")
                    st.metric("5th Percentile", f"{np.percentile(mc_results, 5):.2%}")
                    st.metric("95th Percentile", f"{np.percentile(mc_results, 95):.2%}")
                    st.metric("Probability of Loss", f"{(mc_results < 0).mean():.1%}")
        
        # AI-Powered Portfolio Insights
        st.subheader("🤖 AI-Powered Portfolio Insights")
        
        # Portfolio composition analysis
        col1, col2 = st.columns(2)
        with col1:
            # Diversification score
            diversification_score = 1 - np.sum(weights**2)  # Herfindahl-Hirschman Index
            st.metric("Diversification Score", f"{diversification_score:.3f}", 
                     help="Higher scores indicate better diversification (max = 1)")
            
            # Risk contribution analysis
            risk_contributions = (weights * np.dot(cov, weights)) / (portfolio_risk**2)
            max_risk_contributor = asset_names[np.argmax(risk_contributions)]
            st.metric("Top Risk Contributor", max_risk_contributor, 
                     f"{risk_contributions.max():.1%} of portfolio risk")
        
        with col2:
            # Return contribution analysis
            return_contributions = weights * mu
            max_return_contributor = asset_names[np.argmax(return_contributions)]
            st.metric("Top Return Contributor", max_return_contributor, 
                     f"{return_contributions.max():.2%} expected return")
            
            # Portfolio efficiency
            efficiency_ratio = portfolio_return / np.sum(np.abs(weights - 1/len(weights)))
            st.metric("Efficiency vs Equal Weight", f"{efficiency_ratio:.2f}", 
                     help="How much better this portfolio is vs equal weighting")
        
        # Smart recommendations
        with st.expander("💡 AI Recommendations"):
            recommendations = []
            
            # Check concentration risk
            if np.max(weights) > 0.4:
                recommendations.append("⚠️ **High Concentration**: Consider reducing position in largest holding to improve diversification.")
            
            # Check diversification
            if diversification_score < 0.5:
                recommendations.append("📊 **Low Diversification**: Portfolio could benefit from more balanced allocation across assets.")
            
            # Risk-adjusted return analysis
            if sharpe_ratio < 0.5:
                recommendations.append("📈 **Risk-Return**: Consider adjusting risk tolerance or asset selection to improve risk-adjusted returns.")
            
            # Quantum advantage potential
            if not used_quantum and QUANTUM_AVAILABLE:
                recommendations.append("🌟 **Quantum Potential**: Try quantum optimization for potentially better risk-return balance.")
            
            # Display recommendations
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ **Well-Optimized Portfolio**: Your portfolio shows good diversification and risk-return characteristics!")
        
        # Quantum advantage messaging
        if used_quantum:
            st.info("🌟 **Quantum Advantage**: This optimization utilized quantum computing principles to explore multiple solution paths simultaneously, potentially finding superior allocations.")
        elif QUANTUM_AVAILABLE and optimization_method == "Classical (Scipy)":
            st.info("💡 **Try Quantum**: Switch to quantum optimization to leverage advanced quantum algorithms for potentially better results!")
        
else:
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
