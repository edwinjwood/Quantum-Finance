"""
Application Configuration
Centralized configuration management for the Quantum Portfolio Optimizer.
"""

from dataclasses import dataclass
from typing import Dict, Any
import streamlit as st

@dataclass
class AppConfig:
    """Application configuration settings"""
    
    # App metadata
    APP_TITLE: str = "🚀 Quantum Portfolio Optimization & Diversification"
    PAGE_TITLE: str = "Quantum Portfolio Optimizer"
    
    # Default values
    DEFAULT_TICKERS: str = "AAPL,GOOGL,MSFT,AMZN"
    DEFAULT_RISK_FACTOR: float = 0.5
    DEFAULT_LOOKBACK_DAYS: int = 252  # 1 year
    
    # Monte Carlo settings
    MC_SIMULATIONS: int = 1000
    MC_TIME_HORIZON: int = 252
    
    # Risk analysis settings
    VAR_CONFIDENCE_LEVELS: tuple = (0.05, 0.01)  # 95%, 99%
    
    # UI settings
    CHART_HEIGHT: int = 400
    MAX_ASSETS: int = 10
    MIN_ASSETS: int = 2
    
    # Quantum settings
    QUANTUM_MAX_ITERATIONS: int = 500
    QUANTUM_DEFAULT_ITERATIONS: int = 200
    
    @staticmethod
    def get_optimization_methods() -> list:
        """Get available optimization methods based on quantum availability"""
        try:
            from qiskit_finance.applications.optimization import PortfolioOptimization
            from qiskit_optimization.algorithms import CobylaOptimizer
            return ["Classical (Scipy)", "Quantum (Qiskit)", "Compare Both"]
        except ImportError:
            return ["Classical (Scipy)"]
    
    @staticmethod
    def is_quantum_available() -> bool:
        """Check if quantum computing libraries are available"""
        try:
            from qiskit_finance.applications.optimization import PortfolioOptimization
            from qiskit_optimization.algorithms import CobylaOptimizer
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_quantum_algorithms() -> list:
        """Get available quantum algorithms"""
        return [
            "COBYLA (Classical-Quantum Hybrid)",
            "QAOA (Future)",
            "VQE (Future)"
        ]
