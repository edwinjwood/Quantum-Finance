"""
Risk Analysis Module
Advanced risk analysis functions for portfolio evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

class RiskAnalyzer:
    """Advanced risk analysis for portfolio optimization"""
    
    def analyze_portfolio(self, weights: np.ndarray, returns_data: Optional[pd.DataFrame], 
                         asset_names: List[str]) -> Optional[Dict[str, Any]]:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data (None for demo data)
            asset_names: List of asset names
            
        Returns:
            Dictionary containing risk analysis results or None if no real data
        """
        if returns_data is None:
            return None
        
        # Calculate VaR and CVaR
        var_95, cvar_95 = self.calculate_var_cvar(returns_data.values, weights, 0.05)
        var_99, cvar_99 = self.calculate_var_cvar(returns_data.values, weights, 0.01)
        
        # Calculate Maximum Drawdown
        max_drawdown = self.calculate_maximum_drawdown(returns_data.values, weights)
        
        # Correlation matrix
        correlation_matrix = returns_data.corr()
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'var_99': var_99,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'correlation_matrix': correlation_matrix,
            'asset_names': asset_names
        }
    
    @staticmethod
    def calculate_var_cvar(returns: np.ndarray, weights: np.ndarray, confidence_level: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk
        
        Args:
            returns: Historical returns matrix
            weights: Portfolio weights
            confidence_level: Confidence level (default 0.05 for 95% VaR)
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        portfolio_returns = np.dot(returns, weights)
        var = np.percentile(portfolio_returns, confidence_level * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return var, cvar
    
    @staticmethod
    def calculate_maximum_drawdown(returns: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate maximum drawdown for the portfolio
        
        Args:
            returns: Historical returns matrix
            weights: Portfolio weights
            
        Returns:
            Maximum drawdown as a float
        """
        portfolio_returns = np.dot(returns, weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate running maximum using numpy
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown
    
    @staticmethod
    def monte_carlo_simulation(mu: np.ndarray, cov: np.ndarray, weights: np.ndarray, 
                              num_simulations: int = 1000, time_horizon: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio performance
        
        Args:
            mu: Expected returns vector
            cov: Covariance matrix
            weights: Portfolio weights
            num_simulations: Number of simulations to run
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary containing Monte Carlo simulation results
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate random returns
        simulated_returns = np.random.multivariate_normal(
            mu/252, cov/252, (num_simulations, time_horizon)
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_simulations = []
        for sim in simulated_returns:
            portfolio_returns = np.dot(sim, weights)
            cumulative_return = (1 + portfolio_returns).prod() - 1
            portfolio_simulations.append(cumulative_return)
        
        results = np.array(portfolio_simulations)
        
        return {
            'simulations': results,
            'expected_return': np.mean(results),
            'std_deviation': np.std(results),
            'percentile_5': np.percentile(results, 5),
            'percentile_95': np.percentile(results, 95),
            'probability_of_loss': (results < 0).mean(),
            'num_simulations': num_simulations,
            'time_horizon': time_horizon
        }
