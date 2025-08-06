#!/usr/bin/env python3
"""
Quick test for the COBYLA optimizer fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_quantum_optimizer():
    """Test that the quantum optimizer works without the maxiter error"""
    try:
        from src.optimization.portfolio_optimizer import PortfolioOptimizer
        import numpy as np
        
        # Create simple test data
        np.random.seed(42)
        n_assets = 3
        mu = np.array([0.1, 0.12, 0.15])  # Expected returns
        
        # Create a simple covariance matrix
        cov = np.array([[0.04, 0.01, 0.02],
                        [0.01, 0.05, 0.015],
                        [0.02, 0.015, 0.06]])
        
        optimizer = PortfolioOptimizer()
        
        # Test quantum optimization
        print("Testing quantum optimization...")
        quantum_settings = {'algorithm': 'COBYLA', 'iterations': 100}
        
        result = optimizer.optimize_portfolio(
            mu=mu, 
            cov=cov, 
            method="Quantum (Qiskit)", 
            risk_factor=1.0,
            quantum_settings=quantum_settings
        )
        
        print(f"Quantum optimization result: {result['success']}")
        print(f"Method used: {result['method_used']}")
        print(f"Used quantum: {result['used_quantum']}")
        if result['success']:
            print(f"Weights: {result['weights']}")
            print(f"Portfolio return: {result['portfolio_return']:.4f}")
            print(f"Portfolio risk: {result['portfolio_risk']:.4f}")
        
        return result['success']
        
    except Exception as e:
        print(f"Error in quantum optimization test: {e}")
        return False

if __name__ == "__main__":
    print("Testing COBYLA optimizer fix...")
    success = test_quantum_optimizer()
    
    if success:
        print("✅ Quantum optimization test PASSED!")
    else:
        print("❌ Quantum optimization test FAILED!")
