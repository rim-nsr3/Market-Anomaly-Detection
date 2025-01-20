import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.optimize import minimize

class PortfolioArchitect:
    def __init__(self):
        self.allocations = {
            'defensive': {
                'SPY': 0.15,    # S&P 500
                'QQQ': 0.05,    # Tech
                'GLD': 0.40,    # Gold
                'TLT': 0.30,    # Long-term Bonds
                'Cash': 0.10    # Cash
            },
            'conservative': {
                'SPY': 0.30,
                'QQQ': 0.10,
                'GLD': 0.30,
                'TLT': 0.20,
                'Cash': 0.10
            },
            'moderate': {
                'SPY': 0.45,
                'QQQ': 0.25,
                'GLD': 0.15,
                'TLT': 0.10,
                'Cash': 0.05
            },
            'aggressive': {
                'SPY': 0.60,
                'QQQ': 0.30,
                'GLD': 0.05,
                'TLT': 0.05,
                'Cash': 0.00
            }
        }
    
    def optimize_portfolio(self, market_data: pd.DataFrame, 
                         features: pd.DataFrame) -> Dict:
        """Optimize portfolio based on market conditions"""
        # Get latest market conditions
        latest_features = features.iloc[-1]
        crash_prob = latest_features['crash_probability']
        stress_level = latest_features['stress_index']
        regime = latest_features['regime']
        
        # Select base allocation based on conditions
        base_allocation = self._select_base_allocation(crash_prob, stress_level, regime)
        
        # Calculate optimal weights using risk-adjusted optimization
        optimal_weights = self._optimize_weights(market_data, base_allocation, stress_level)
        
        # Generate implementation plan
        implementation_steps = self._generate_implementation_plan(
            optimal_weights, 
            base_allocation, 
            crash_prob, 
            stress_level
        )
        
        return {
            'regime': 'bearish' if regime < 0 else 'neutral' if regime == 0 else 'bullish',
            'stress_level': stress_level,
            'optimal_weights': optimal_weights,
            'base_allocation': base_allocation,
            'implementation': implementation_steps
        }
    
    def _select_base_allocation(self, crash_prob: float, 
                              stress_level: float, 
                              regime: int) -> Dict:
        """Select appropriate base allocation"""
        if crash_prob > 0.7 or stress_level > 1.5:
            return self.allocations['defensive']
        elif crash_prob > 0.4 or stress_level > 1.0:
            return self.allocations['conservative']
        elif crash_prob > 0.2 or regime == 0:
            return self.allocations['moderate']
        else:
            return self.allocations['aggressive']
    
    def _optimize_weights(self, market_data: pd.DataFrame,
                         base_allocation: Dict,
                         stress_level: float) -> Dict:
        """Optimize portfolio weights using risk-adjusted optimization"""
        # Calculate returns for available assets
        returns = pd.DataFrame()
        for asset in base_allocation.keys():
            if asset != 'Cash' and f'{asset}_close' in market_data.columns:
                returns[asset] = market_data[f'{asset}_close'].pct_change()
        
        # Add risk-free rate for cash
        returns['Cash'] = 0.02 / 252  # Assuming 2% annual risk-free rate
        
        returns = returns.dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Define optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Risk tolerance decreases as stress increases
        risk_tolerance = 1.0 / (1.0 + stress_level)
        
        # Objective function: Maximize Sharpe Ratio with stress adjustment
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            stress_penalty = stress_level * np.sum(np.abs(weights - list(base_allocation.values())))
            return -(portfolio_return - risk_tolerance * portfolio_std - stress_penalty)
        
        # Initial guess (base allocation)
        initial_weights = list(base_allocation.values())
        bounds = [(0.05, 0.6) for _ in range(len(base_allocation))]
        
        try:
            # Run optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return dict(zip(base_allocation.keys(), result.x))
            else:
                return base_allocation
                
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return base_allocation
    
    def _calculate_portfolio_stats(self, weights: Dict, 
                                 market_data: pd.DataFrame) -> Dict:
        """Calculate portfolio statistics"""
        returns = pd.DataFrame()
        for asset, weight in weights.items():
            if asset != 'Cash' and f'{asset}_close' in market_data.columns:
                returns[asset] = market_data[f'{asset}_close'].pct_change() * weight
        
        portfolio_returns = returns.sum(axis=1)
        
        return {
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()
    
    def _generate_implementation_plan(self, optimal_weights: Dict,
                                   base_allocation: Dict,
                                   crash_prob: float,
                                   stress_level: float) -> List[str]:
        """Generate detailed implementation plan"""
        plan = []
        
        # Risk level assessment
        if crash_prob > 0.7:
            plan.append("⚠️ High crash probability detected - implementing defensive positioning")
        elif crash_prob > 0.4:
            plan.append("⚠️ Elevated crash risk - maintaining conservative allocation")
        else:
            plan.append("✅ Low crash risk - optimizing for growth")
        
        # Asset allocation changes
        plan.append("\n📊 Target Allocation:")
        for asset, target in optimal_weights.items():
            current = base_allocation[asset]
            diff = target - current
            if abs(diff) >= 0.05:  # Only show significant changes
                direction = "Increase" if diff > 0 else "Reduce"
                plan.append(f"  • {asset}: {direction} by {abs(diff):.1%} to {target:.1%}")
            else:
                plan.append(f"  • {asset}: Maintain at {target:.1%}")
        
        # Implementation timing
        plan.append("\n⌛ Implementation Schedule:")
        if stress_level > 1.5:
            plan.append("  • High market stress - implement changes gradually over 5-10 days")
            plan.append("  • Consider using limit orders and avoiding market impact")
        elif stress_level > 1.0:
            plan.append("  • Elevated stress - implement changes over 3-5 days")
            plan.append("  • Use a mix of limit and market orders")
        else:
            plan.append("  • Normal market conditions - implement changes over 1-2 days")
            plan.append("  • Standard market orders acceptable")
        
        # Risk management recommendations
        plan.append("\n🛡️ Risk Management:")
        if crash_prob > 0.5:
            plan.append("  • Set stop-loss orders at -5% below entry")
            plan.append("  • Consider protective put options")
        if stress_level > 1.2:
            plan.append("  • Increase position size gradually")
            plan.append("  • Maintain higher cash reserves")
        
        return plan