"""
Portfolio Optimization Module
Contains classical and quantum optimization algorithms for portfolio optimization.
"""

import numpy as np
import streamlit as st
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Optional

class PortfolioOptimizer:
    """Portfolio optimization using classical and quantum methods"""
    
    def optimize_portfolio(self, mu: np.ndarray, cov: np.ndarray, method: str, 
                          risk_factor: float, quantum_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main portfolio optimization method
        
        Args:
            mu: Expected returns vector
            cov: Covariance matrix
            method: Optimization method ('Classical (Scipy)', 'Quantum (Qiskit)', 'Compare Both')
            risk_factor: Risk aversion parameter
            quantum_settings: Quantum algorithm settings
            
        Returns:
            Dictionary containing optimization results
        """
        quantum_settings = quantum_settings or {}
        
        if method == "Compare Both" and self._is_quantum_available():
            return self._compare_both_methods(mu, cov, risk_factor, quantum_settings)
        elif method == "Quantum (Qiskit)" and self._is_quantum_available():
            return self._quantum_optimization(mu, cov, risk_factor, quantum_settings)
        else:
            return self._classical_optimization(mu, cov, risk_factor)
    
    def _classical_optimization(self, mu: np.ndarray, cov: np.ndarray, risk_factor: float) -> Dict[str, Any]:
        """Classical mean-variance optimization using scipy"""
        n = len(mu)
        
        # Objective: minimize risk - return + penalty for risk aversion
        def objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            return portfolio_risk * risk_factor - portfolio_return
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess: equal weights
        x0 = np.array([1/n] * n)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
        else:
            weights = np.ones(n) / n  # Fallback to equal weights
        
        return {
            'weights': weights,
            'method_used': 'Classical (Scipy)',
            'used_quantum': False,
            'success': result.success,
            'portfolio_return': np.dot(weights, mu),
            'portfolio_risk': np.sqrt(np.dot(weights, np.dot(cov, weights))),
            'optimization_details': {
                'iterations': result.nit if hasattr(result, 'nit') else None,
                'message': result.message if hasattr(result, 'message') else None
            }
        }
    
    def _quantum_optimization(self, mu: np.ndarray, cov: np.ndarray, risk_factor: float, 
                            quantum_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum portfolio optimization using Qiskit"""
        try:
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.algorithms import CobylaOptimizer
            
            # Create a continuous quadratic program
            qp = QuadraticProgram()
            n = len(mu)
            
            # Add continuous variables for portfolio weights
            for i in range(n):
                qp.continuous_var(name=f'x_{i}', lowerbound=0, upperbound=1)
            
            # Objective: minimize portfolio risk - expected return
            linear_terms = {}
            quadratic_terms = {}
            
            # Linear terms (negative expected returns)
            for i in range(n):
                linear_terms[f'x_{i}'] = -mu[i]
            
            # Quadratic terms (covariance matrix scaled by risk aversion)
            for i in range(n):
                for j in range(n):
                    if i <= j:  # Only upper triangle needed
                        quadratic_terms[(f'x_{i}', f'x_{j}')] = risk_factor * cov[i, j]
            
            qp.minimize(linear=linear_terms, quadratic=quadratic_terms)
            
            # Constraint: sum of weights = 1
            linear_constraint = {}
            for i in range(n):
                linear_constraint[f'x_{i}'] = 1
            qp.linear_constraint(linear=linear_constraint, sense='==', rhs=1, name='budget')
            
            # Quantum optimization details
            quantum_details = {
                'problem_size': n,
                'variables': n,
                'constraints': 1,
                'quadratic_terms': len(quadratic_terms),
                'algorithm': quantum_settings.get('algorithm', 'COBYLA')
            }
            
            # Use COBYLA optimizer with the continuous problem
            max_iterations = quantum_settings.get('iterations', 200)
            try:
                # Try with maxiter parameter (older versions)
                optimizer = CobylaOptimizer(maxiter=max_iterations)
            except TypeError:
                # Fallback for newer versions that don't support maxiter
                optimizer = CobylaOptimizer()
            result = optimizer.solve(qp)
            
            if result.x is not None and len(result.x) == n:
                weights = np.array(result.x)
                # Ensure weights are normalized and non-negative
                weights = np.maximum(weights, 0)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones(n) / n
                
                return {
                    'weights': weights,
                    'method_used': 'Quantum (Qiskit)',
                    'used_quantum': True,
                    'success': True,
                    'portfolio_return': np.dot(weights, mu),
                    'portfolio_risk': np.sqrt(np.dot(weights, np.dot(cov, weights))),
                    'quantum_details': quantum_details,
                    'optimization_details': {
                        'iterations': max_iterations,
                        'quantum_algorithm': quantum_settings.get('algorithm', 'COBYLA')
                    }
                }
            else:
                # Fallback to classical if quantum fails
                st.warning("Quantum optimization failed. Using classical fallback.")
                return self._classical_optimization(mu, cov, risk_factor)
                
        except Exception as e:
            st.warning(f"Quantum optimization encountered an issue: {str(e)}. Using classical fallback.")
            return self._classical_optimization(mu, cov, risk_factor)
    
    def _compare_both_methods(self, mu: np.ndarray, cov: np.ndarray, risk_factor: float, 
                             quantum_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Compare classical and quantum optimization methods"""
        # Run classical optimization
        classical_result = self._classical_optimization(mu, cov, risk_factor)
        
        # Run quantum optimization
        quantum_result = self._quantum_optimization(mu, cov, risk_factor, quantum_settings)
        
        # Calculate Sharpe ratios for comparison
        classical_sharpe = (classical_result['portfolio_return'] / 
                           classical_result['portfolio_risk'] if classical_result['portfolio_risk'] > 0 else 0)
        quantum_sharpe = (quantum_result['portfolio_return'] / 
                         quantum_result['portfolio_risk'] if quantum_result['portfolio_risk'] > 0 else 0)
        
        # Determine winner
        if quantum_sharpe > classical_sharpe:
            primary_result = quantum_result
            winner = "quantum"
        else:
            primary_result = classical_result
            winner = "classical"
        
        return {
            'weights': primary_result['weights'],
            'method_used': 'Compare Both',
            'used_quantum': primary_result['used_quantum'],
            'success': True,
            'portfolio_return': primary_result['portfolio_return'],
            'portfolio_risk': primary_result['portfolio_risk'],
            'comparison_results': {
                'classical': classical_result,
                'quantum': quantum_result,
                'classical_sharpe': classical_sharpe,
                'quantum_sharpe': quantum_sharpe,
                'winner': winner
            },
            'quantum_details': quantum_result.get('quantum_details', {}),
            'optimization_details': primary_result.get('optimization_details', {})
        }
    
    @staticmethod
    def _is_quantum_available() -> bool:
        """Check if quantum computing libraries are available"""
        try:
            from qiskit_finance.applications.optimization import PortfolioOptimization
            from qiskit_optimization.algorithms import CobylaOptimizer
            return True
        except ImportError:
            return False
