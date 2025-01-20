import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class MarketVisualizer:
    def __init__(self, theme: str = 'light'):
        self.theme = 'plotly_white' if theme == 'light' else 'plotly_dark'
        self.colors = {
            'bullish': '#22c55e',  # green
            'neutral': '#f59e0b',  # amber
            'bearish': '#ef4444',  # red
            'baseline': '#3b82f6'  # blue
        }
    
    def plot_market_dashboard(self, market_data: pd.DataFrame, 
                            features: pd.DataFrame) -> go.Figure:
        """Create enhanced market dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Asset Performance', 'Crash Probability',
                'Market Breadth', 'Stress Index',
                'Volume Analysis', 'Regime Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Asset Performance
        for col in ['SPY_close', 'QQQ_close', 'GLD_close', 'TLT_close']:
            if col in market_data.columns:
                normalized = market_data[col] / market_data[col].iloc[0] * 100
                fig.add_trace(
                    go.Scatter(x=normalized.index, y=normalized, 
                              name=col.replace('_close', ''),
                              mode='lines'),
                    row=1, col=1
                )
        
        # 2. Crash Probability
        if 'crash_probability' in features.columns:
            fig.add_trace(
                go.Scatter(x=features.index, y=features['crash_probability'],
                          name='Crash Probability',
                          fill='tozeroy',
                          line=dict(color=self.colors['bearish'])),
                row=1, col=2
            )
            fig.add_hline(y=0.7, line_dash="dash", row=1, col=2,
                         annotation_text="High Risk")
            fig.add_hline(y=0.3, line_dash="dash", row=1, col=2,
                         annotation_text="Low Risk")
        
        # 3. Market Breadth
        if 'market_breadth' in features.columns:
            fig.add_trace(
                go.Scatter(x=features.index, y=features['market_breadth'],
                          name='Market Breadth',
                          fill='tozeroy',
                          line=dict(color=self.colors['baseline'])),
                row=2, col=1
            )
        
        # 4. Stress Index
        if 'stress_index' in features.columns:
            fig.add_trace(
                go.Scatter(x=features.index, y=features['stress_index'],
                          name='Stress Index',
                          line=dict(color=self.colors['bearish'])),
                row=2, col=2
            )
        
        # 5. Volume Analysis
        for col in ['SPY_volume', 'QQQ_volume']:
            if col in market_data.columns:
                normalized_vol = market_data[col] / market_data[col].rolling(20).mean()
                fig.add_trace(
                    go.Scatter(x=normalized_vol.index, y=normalized_vol,
                              name=f'{col.replace("_volume", "")} Vol Ratio',
                              mode='lines'),
                    row=3, col=1
                )
        
        # 6. Regime Analysis
        if 'regime' in features.columns:
            fig.add_trace(
                go.Scatter(x=features.index, y=features['regime'],
                          name='Market Regime',
                          line=dict(color=self.colors['baseline'])),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            template=self.theme,
            title={
                'text': "Market Analysis Dashboard",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=2)
        fig.update_yaxes(title_text="Breadth", row=2, col=1)
        fig.update_yaxes(title_text="Stress Level", row=2, col=2)
        fig.update_yaxes(title_text="Volume Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Regime", row=3, col=2)
        
        return fig
    
    def plot_portfolio_analysis(self, portfolio: dict,
                              market_data: pd.DataFrame) -> go.Figure:
        """Create enhanced portfolio analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Asset Allocation', 'Asset Performance',
                'Risk Analysis', 'Rebalancing Needs'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Asset Allocation Pie Chart
        fig.add_trace(
            go.Pie(
                labels=list(portfolio['optimal_weights'].keys()),
                values=list(portfolio['optimal_weights'].values()),
                hole=0.3,
                name="Allocation"
            ),
            row=1, col=1
        )
        
        # 2. Asset Performance
        if 'SPY_close' in market_data.columns:
            returns = market_data['SPY_close'].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=cumulative_returns,
                    name='Portfolio Returns',
                    line=dict(color=self.colors['baseline'])
                ),
                row=1, col=2
            )
        
        # 3. Risk Analysis
        risk_data = {
            'Asset': list(portfolio['optimal_weights'].keys()),
            'Weight': list(portfolio['optimal_weights'].values())
        }
        fig.add_trace(
            go.Bar(
                x=risk_data['Asset'],
                y=risk_data['Weight'],
                name="Asset Weights"
            ),
            row=2, col=1
        )
        
        # 4. Rebalancing Needs
        if 'base_allocation' in portfolio:
            base = portfolio['base_allocation']
            optimal = portfolio['optimal_weights']
            assets = list(optimal.keys())
            rebalance = [optimal[a] - base.get(a, 0) for a in assets]
            
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=rebalance,
                    name="Rebalancing Needed",
                    marker_color=np.where(np.array(rebalance) > 0, 
                                        self.colors['bullish'], 
                                        self.colors['bearish'])
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            template=self.theme,
            showlegend=True,
            title={
                'text': "Portfolio Analysis Dashboard",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        return fig