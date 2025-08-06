"""
Data Management Module
Handles data fetching, processing, and validation for portfolio optimization.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

class DataManager:
    """Manages data fetching and processing for portfolio optimization"""
    
    @staticmethod
    @st.cache_data
    def fetch_stock_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch stock data and calculate returns and covariance
        
        Returns:
            Tuple of (expected_returns, covariance_matrix, returns_data, price_data)
        """
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
            return DataManager._try_alternative_fetch(tickers, start_date, end_date)
    
    @staticmethod
    def _try_alternative_fetch(tickers: List[str], start_date: datetime, end_date: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Alternative data fetching method"""
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
    
    @staticmethod
    def generate_demo_data(num_assets: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic demo data for testing
        
        Returns:
            Tuple of (expected_returns, covariance_matrix, asset_names)
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate expected returns
        mu = np.random.uniform(0.05, 0.2, num_assets)
        
        # Generate covariance matrix
        sigma = np.random.uniform(0.01, 0.05, (num_assets, num_assets))
        cov = np.dot(sigma, sigma.T)
        
        # Asset names
        asset_names = [f"Asset {i+1}" for i in range(num_assets)]
        
        return mu, cov, asset_names
    
    def get_portfolio_data(self, data_source: str, tickers: List[str] = None, 
                          start_date: datetime = None, end_date: datetime = None, 
                          num_assets: int = 4) -> Optional[Dict[str, Any]]:
        """
        Get portfolio data based on source type
        
        Returns:
            Dictionary containing portfolio data or None if failed
        """
        if data_source == "Real Stock Data" and tickers and len(tickers) > 1:
            # Fetch real stock data
            mu, cov, returns_data, price_data = self.fetch_stock_data(tickers, start_date, end_date)
            
            if mu is not None and cov is not None:
                st.success(f"✅ Successfully loaded data for {len(tickers)} stocks from {start_date} to {end_date}")
                return {
                    'expected_returns': mu,
                    'covariance_matrix': cov,
                    'returns_data': returns_data,
                    'price_data': price_data,
                    'asset_names': tickers,
                    'data_source': 'real'
                }
            else:
                st.error("Failed to load stock data. Using demo data instead.")
                # Fallback to demo data
                mu, cov, asset_names = self.generate_demo_data(num_assets)
                return {
                    'expected_returns': mu,
                    'covariance_matrix': cov,
                    'returns_data': None,
                    'price_data': None,
                    'asset_names': asset_names,
                    'data_source': 'demo'
                }
        
        elif data_source == "Real Stock Data" and (not tickers or len(tickers) <= 1):
            st.warning("Please enter at least 2 stock tickers for portfolio optimization.")
            # Use demo data
            mu, cov, asset_names = self.generate_demo_data(4)
            return {
                'expected_returns': mu,
                'covariance_matrix': cov,
                'returns_data': None,
                'price_data': None,
                'asset_names': asset_names,
                'data_source': 'demo'
            }
        
        else:
            # Demo data
            mu, cov, asset_names = self.generate_demo_data(num_assets)
            return {
                'expected_returns': mu,
                'covariance_matrix': cov,
                'returns_data': None,
                'price_data': None,
                'asset_names': asset_names,
                'data_source': 'demo'
            }
