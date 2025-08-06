"""
AI-Powered Portfolio Insights Module
Generates intelligent insights and recommendations for portfolio optimization.
"""

import numpy as np
from typing import Dict, Any, List

class AIInsights:
    """AI-powered portfolio analysis and recommendations"""
    
    def generate_insights(self, optimization_result: Dict[str, Any], 
                         data_result: Dict[str, Any], user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive AI-powered portfolio insights
        
        Args:
            optimization_result: Results from portfolio optimization
            data_result: Portfolio data and metadata
            user_inputs: User configuration settings
            
        Returns:
            Dictionary containing AI insights and recommendations
        """
        weights = optimization_result['weights']
        mu = data_result['expected_returns']
        cov = data_result['covariance_matrix']
        asset_names = data_result['asset_names']
        
        # Calculate portfolio metrics
        portfolio_return = optimization_result['portfolio_return']
        portfolio_risk = optimization_result['portfolio_risk']
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Calculate insights
        diversification_score = self._calculate_diversification_score(weights)
        risk_contributions = self._calculate_risk_contributions(weights, cov, portfolio_risk)
        return_contributions = self._calculate_return_contributions(weights, mu)
        efficiency_ratio = self._calculate_efficiency_ratio(weights, portfolio_return)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            weights, diversification_score, sharpe_ratio, 
            optimization_result.get('used_quantum', False), user_inputs
        )
        
        return {
            'diversification_score': diversification_score,
            'risk_contributions': risk_contributions,
            'return_contributions': return_contributions,
            'efficiency_ratio': efficiency_ratio,
            'top_risk_contributor': asset_names[np.argmax(risk_contributions)],
            'top_return_contributor': asset_names[np.argmax(return_contributions)],
            'recommendations': recommendations,
            'portfolio_metrics': {
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio
            }
        }
    
    @staticmethod
    def _calculate_diversification_score(weights: np.ndarray) -> float:
        """Calculate diversification score using Herfindahl-Hirschman Index"""
        return 1 - np.sum(weights**2)
    
    @staticmethod
    def _calculate_risk_contributions(weights: np.ndarray, cov: np.ndarray, portfolio_risk: float) -> np.ndarray:
        """Calculate risk contributions of each asset"""
        if portfolio_risk == 0:
            return np.zeros_like(weights)
        return (weights * np.dot(cov, weights)) / (portfolio_risk**2)
    
    @staticmethod
    def _calculate_return_contributions(weights: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Calculate return contributions of each asset"""
        return weights * mu
    
    @staticmethod
    def _calculate_efficiency_ratio(weights: np.ndarray, portfolio_return: float) -> float:
        """Calculate portfolio efficiency vs equal weight benchmark"""
        equal_weight_deviation = np.sum(np.abs(weights - 1/len(weights)))
        if equal_weight_deviation == 0:
            return 1.0
        return portfolio_return / equal_weight_deviation
    
    def _generate_recommendations(self, weights: np.ndarray, diversification_score: float, 
                                 sharpe_ratio: float, used_quantum: bool, 
                                 user_inputs: Dict[str, Any]) -> List[str]:
        """Generate AI-powered portfolio recommendations"""
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
        elif sharpe_ratio > 1.5:
            recommendations.append("🎯 **Excellent Risk-Return**: Portfolio shows strong risk-adjusted performance!")
        
        # Quantum advantage potential
        if not used_quantum and self._is_quantum_available():
            recommendations.append("🌟 **Quantum Potential**: Try quantum optimization for potentially better risk-return balance.")
        
        # Risk tolerance suggestions
        risk_factor = user_inputs.get('risk_factor', 0.5)
        if risk_factor < 0.3:
            recommendations.append("🛡️ **Conservative Approach**: Your low risk aversion may benefit from more aggressive growth assets.")
        elif risk_factor > 0.7:
            recommendations.append("⚡ **Aggressive Strategy**: Consider adding some defensive assets to balance risk.")
        
        # Portfolio size recommendations
        num_assets = len(weights)
        if num_assets < 5:
            recommendations.append("🔄 **Diversification Opportunity**: Consider adding more assets to improve portfolio diversification.")
        elif num_assets > 15:
            recommendations.append("🎯 **Complexity Management**: Portfolio may benefit from consolidating to fewer, higher-conviction positions.")
        
        # Default positive message if no issues found
        if not recommendations:
            recommendations.append("✅ **Well-Optimized Portfolio**: Your portfolio shows good diversification and risk-return characteristics!")
        
        return recommendations
    
    @staticmethod
    def _is_quantum_available() -> bool:
        """Check if quantum computing libraries are available"""
        try:
            from qiskit_finance.applications.optimization import PortfolioOptimization
            from qiskit_optimization.algorithms import CobylaOptimizer
            return True
        except ImportError:
            return False
