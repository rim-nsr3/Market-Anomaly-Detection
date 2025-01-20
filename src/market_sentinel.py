import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MarketSentinel:
    def __init__(self):
        self.tickers = {
            'market': ['SPY', 'QQQ', 'IWM'],  # Major indices
            'sectors': ['XLF', 'XLK', 'XLE'],  # Key sectors
            'safe_haven': ['GLD', 'TLT'],      # Safe havens
            'volatility': ['^VIX']             # Volatility
        }
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.crash_thresholds = {
            'short_term': (-0.05, 5),   # 5% in 5 days
            'medium_term': (-0.10, 20),  # 10% in 20 days
        }
    
    def fetch_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch market data with robust error handling"""
        try:
            # Extend start date to ensure enough training data
            start_date_obj = pd.to_datetime(start_date)
            extended_start = (start_date_obj - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            
            print(f"Fetching data from {extended_start} to {end_date}")
            
            # Initialize empty DataFrame for market data
            market_data = pd.DataFrame()
            
            # Get list of all tickers
            all_tickers = [ticker for sublist in self.tickers.values() for ticker in sublist]
            
            # Fetch data for each ticker
            for ticker in all_tickers:
                try:
                    # Download data
                    ticker_data = yf.download(ticker, 
                                           start=extended_start, 
                                           end=end_date, 
                                           progress=False)
                    
                    if not ticker_data.empty and len(ticker_data) > 100:
                        # If successful, add Close price
                        market_data[f'{ticker}_close'] = ticker_data['Close']
                        
                        # Add Volume if available (some assets like indices don't have volume)
                        if 'Volume' in ticker_data.columns:
                            market_data[f'{ticker}_volume'] = ticker_data['Volume']
                            
                        print(f"Successfully fetched {ticker} with {len(ticker_data)} samples")
                    else:
                        print(f"No data or insufficient samples for {ticker}")
                
                except Exception as e:
                    print(f"Error fetching {ticker}: {str(e)}")
                    continue
            
            # Verify we have enough data
            if market_data.empty:
                raise ValueError("Failed to fetch any market data")
            
            # Handle missing data
            market_data = market_data.fillna(method='ffill').fillna(method='bfill')
            
            # Verify we have the minimum required tickers (SPY is essential)
            if 'SPY_close' not in market_data.columns:
                raise ValueError("Failed to fetch essential ticker: SPY")
            
            print(f"Successfully fetched data with shape: {market_data.shape}")
            
            # Trim to requested date range
            market_data = market_data[market_data.index >= start_date_obj]
            
            return market_data
            
        except Exception as e:
            print(f"Critical error in market data fetching: {str(e)}")
            raise ValueError(f"Failed to fetch market data: {str(e)}")
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels with better data handling"""
        try:
            print(f"Starting feature preparation with {len(df)} samples")
            features = pd.DataFrame(index=df.index)
            
            # Basic features using SPY (main index)
            spy_close = df['SPY_close']
            features['spy_returns'] = spy_close.pct_change()
            features['spy_volatility'] = features['spy_returns'].rolling(window=20, min_periods=10).std()
            
            # Simple moving averages
            features['spy_sma50'] = spy_close.rolling(window=50, min_periods=25).mean() / spy_close
            features['spy_sma200'] = spy_close.rolling(window=200, min_periods=100).mean() / spy_close
            
            # Add QQQ features if available
            if 'QQQ_close' in df.columns:
                qqq_close = df['QQQ_close']
                features['qqq_returns'] = qqq_close.pct_change()
                features['qqq_volatility'] = features['qqq_returns'].rolling(window=20, min_periods=10).std()
            
            # Forward fill any missing values
            features = features.fillna(method='ffill')
            
            # Create crash labels using SPY
            labels = pd.Series(0, index=df.index)
            forward_returns = spy_close.pct_change(20).shift(-20)
            labels = (forward_returns < -0.10).astype(int)  # 10% drop in 20 days
            
            # Remove any remaining NaN values
            features = features.dropna()
            labels = labels[features.index]
            
            print(f"Completed feature preparation with {len(features)} samples and {len(features.columns)} features")
            
            if len(features) == 0:
                return pd.DataFrame(), pd.Series()
                
            return features, labels
            
        except Exception as e:
            print(f"Error in feature preparation: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_model(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """Train model with simplified approach and better error handling"""
        try:
            if features.empty or labels.empty:
                raise ValueError("Empty features or labels provided")
                
            print(f"Training model with {len(features)} samples")
            
            # Scale features
            X = self.scaler.fit_transform(features)
            y = labels
            
            # Train model
            self.model.fit(X, y)
            
            # Get predictions
            y_pred = self.model.predict(X)
            
            # Get probabilities
            y_prob = self.model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                'classification_report': classification_report(y, y_pred),
                'feature_importance': dict(zip(features.columns, 
                                            self.model.feature_importances_)),
                'accuracy': (y == y_pred).mean(),
                'predictions': pd.Series(y_prob, index=features.index)
            }
            
            print(f"Model training completed. Accuracy: {metrics['accuracy']:.2f}")
            return metrics
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            # Return default metrics
            return {
                'classification_report': 'Training failed',
                'feature_importance': {col: 0.0 for col in features.columns},
                'accuracy': 0.0,
                'predictions': pd.Series(0.5, index=features.index)
            }
    
    def predict_crash_probability(self, features: pd.DataFrame) -> pd.Series:
        """Predict crash probabilities with error handling"""
        try:
            if features.empty:
                raise ValueError("Empty features provided")
                
            # Scale features using the same scaler used in training
            features_scaled = self.scaler.transform(features)
            
            # Get probability predictions
            probabilities = self.model.predict_proba(features_scaled)[:, 1]
            
            # Return as a series with the original index
            return pd.Series(probabilities, index=features.index)
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            # Return default probabilities
            return pd.Series(0.5, index=features.index)
    
    def calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market features with simple, robust indicators"""
        try:
            market_features = pd.DataFrame(index=df.index)
            
            # Basic market indicators
            spy_returns = df['SPY_close'].pct_change()
            market_features['trend'] = spy_returns.rolling(window=20, min_periods=10).mean()
            market_features['volatility'] = spy_returns.rolling(window=20, min_periods=10).std()
            
            # Market regime (simple moving average based)
            sma50 = df['SPY_close'].rolling(window=50, min_periods=25).mean()
            sma200 = df['SPY_close'].rolling(window=200, min_periods=100).mean()
            market_features['regime'] = np.where(sma50 > sma200, 1, -1)
            
            # Calculate market breadth (percentage of assets above their moving averages)
            breadth_indicators = []
            for col in df.columns:
                if '_close' in col:
                    price = df[col]
                    ma50 = price.rolling(window=50, min_periods=25).mean()
                    breadth_indicators.append(price > ma50)
            
            if breadth_indicators:
                market_features['market_breadth'] = pd.concat(breadth_indicators, axis=1).mean(axis=1)
            else:
                market_features['market_breadth'] = 0.5  # Default value if no indicators available
            
            # Fill any missing values
            market_features = market_features.fillna(method='ffill')
            
            # Crash probability (will be filled by ML model)
            market_features['crash_probability'] = 0.5  # Default value
            
            # Stress index
            market_features['stress_index'] = (
                -market_features['trend'] + 
                market_features['volatility'] + 
                (0.5 - market_features['market_breadth'])  # Add market breadth component
            ).fillna(0)
            
            return market_features
            
        except Exception as e:
            print(f"Error in market features calculation: {str(e)}")
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['trend', 'volatility', 'regime', 
                                    'crash_probability', 'stress_index',
                                    'market_breadth'])
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Generate market analysis"""
        features = self.calculate_market_features(df)
        latest = features.iloc[-1]
        
        return {
            'crash_probability': latest['crash_probability'],
            'market_regime': 'bearish' if latest['regime'] == -1 else
                           'neutral' if latest['regime'] == 0 else 'bullish',
            'stress_level': latest['stress_index'],
            'key_indicators': {
                'trend': latest['trend'],
                'volatility': latest['volatility'],
                'market_breadth': latest['market_breadth']
            },
            'insights': self._generate_insights(features)
        }
    
    def _generate_insights(self, features: pd.DataFrame) -> List[str]:
        """Generate market insights"""
        insights = []
        latest = features.iloc[-1]
        
        # Crash probability insights
        prob = latest['crash_probability']
        if prob > 0.7:
            insights.append("⚠️ High crash risk detected - consider defensive positioning")
        elif prob > 0.3:
            insights.append("⚠️ Elevated crash risk - maintain balanced exposure")
        else:
            insights.append("✅ Low crash risk - maintain strategic allocation")
        
        # Trend insights
        if latest['trend'] < -0.02:
            insights.append("📉 Downtrend detected - focus on quality and defense")
        elif latest['trend'] > 0.02:
            insights.append("📈 Uptrend confirmed - maintain growth exposure")
        
        # Breadth insights
        if latest['market_breadth'] < 0.4:
            insights.append("⚠️ Poor market breadth - narrow leadership")
        elif latest['market_breadth'] > 0.7:
            insights.append("✅ Strong market breadth - broad participation")
        
        return insights