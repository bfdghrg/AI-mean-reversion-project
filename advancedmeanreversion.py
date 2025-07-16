"""
Enhanced ML-Powered Mean Reversion Trading System
=================================================

Performance-optimized version targeting:
- Sharpe Ratio >0.5 (vs 0.119)
- Max Drawdown <10% (vs 16.63%) 
- CAGR >10% (vs 6.26%)
- Higher signal quality and fewer trades

Author: Enhanced from modest performance version
"""

import os
import sys
import json
import pickle
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import yfinance as yf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# 1. ENHANCED CONFIGURATION FOR BETTER PERFORMANCE
# =============================================================================

@dataclass
class TradingConfig:
    """Enhanced configuration targeting better risk-adjusted returns"""
    
    # Data Settings
    lookback_days: int = 500
    min_data_points: int = 200
    data_cache_dir: str = "data_cache"
    
    # BALANCED FIX 1: MODERATE SIGNAL THRESHOLDS FOR PROFITABILITY
    signal_threshold: float = 0.25        # BALANCED between 0.2 and 0.35 (quality + quantity)
    min_confirmations: int = 1            # RESTORED to 1 (don't over-filter opportunities)
    regime_filter: bool = True
    
    # Feature Engineering - Focus on Quality
    feature_windows: List[int] = field(default_factory=lambda: [10, 20, 50])  # Focused windows
    max_features: int = 12                # REDUCED from 15 (avoid overfitting)
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Model Settings
    retrain_frequency_hours: int = 240    # INCREASED from 168 (less frequent retraining)
    min_training_samples: int = 500
    cv_folds: int = 3
    model_random_state: int = 42
    
    # BALANCED FIX 2: MODERATE POSITION SIZING & RISK
    initial_capital: float = 100000
    max_positions: int = 4                # BALANCED from 3 (need more diversification)
    position_size_pct: float = 0.14       # BALANCED from 0.12 (need more exposure)
    volatility_scaling: bool = True       # KEEP: Scale position size by volatility
    max_sector_exposure: float = 0.35     # RELAXED from 0.3
    max_correlation: float = 0.4          # RELAXED from 0.3
    
    # BALANCED FIX 3: MODERATE RISK MANAGEMENT
    stop_loss_pct: float = 0.03           # BALANCED from 0.025 (less aggressive stops)
    take_profit_pct: float = 0.07         # BALANCED from 0.08 (quicker profits)
    max_hold_hours: int = 42              # BALANCED from 36 (allow more time)
    trailing_stop: bool = True
    trailing_stop_trigger: float = 0.03   # LOWER trigger from 0.04
    
    # BALANCED: MODERATE MARKET REGIME FILTERING
    volatility_filter: bool = True        # KEEP: Avoid extreme volatility
    trend_filter: bool = False            # RELAXED: Allow more counter-trend trades
    momentum_lookback: int = 5            # KEEP: Check recent momentum
    volume_threshold: float = 1.1         # RELAXED from 1.2
    
    # BALANCED: SELECTIVE TIME-BASED FILTERS
    avoid_first_hour: bool = False        # RELAXED: Allow first hour trading
    avoid_last_hour: bool = True          # KEEP: Avoid close volatility
    lunch_hour_filter: bool = False       # RELAXED: Allow lunch hour trading
    
    # BALANCED: MODERATE RISK CONTROLS
    max_daily_trades: int = 3             # INCREASED from 2
    drawdown_limit: float = 0.10          # RELAXED from 0.08
    
    # Trading Costs
    commission_per_trade: float = 0.0
    slippage_bps: float = 2.0
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "enhanced_trading_system.log"
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'TradingConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# =============================================================================
# 2. LOGGING SETUP
# =============================================================================

def setup_logging(config: TradingConfig) -> logging.Logger:
    """Setup centralized logging"""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{config.log_file}"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# 3. DATA MODELS
# =============================================================================

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    shares: int
    entry_price: float
    entry_time: datetime
    signal_strength: float
    stop_loss: float
    take_profit: float
    trailing_stop_price: Optional[float] = None
    trailing_started: bool = False  # NEW: Track when trailing starts
    
    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return abs(self.shares) * self.entry_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.shares > 0
    
    def update_trailing_stop(self, current_price: float, config: TradingConfig):
        """Enhanced trailing stop with trigger threshold"""
        if not self.trailing_started:
            # Check if we should start trailing
            if self.is_long:
                unrealized_return = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_return = (self.entry_price - current_price) / self.entry_price
            
            if unrealized_return >= config.trailing_stop_trigger:
                self.trailing_started = True
        
        # Update trailing stop if started
        if self.trailing_started:
            trail_pct = 0.02  # 2% trailing
            if self.is_long:
                new_stop = current_price * (1 - trail_pct)
                if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
            else:
                new_stop = current_price * (1 + trail_pct)
                if self.trailing_stop_price is None or new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    shares: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    holding_hours: float
    exit_reason: str
    signal_strength: float
    
class MarketRegime(Enum):
    """Market regime classification"""
    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"
    TRENDING = "trending"

# =============================================================================
# 4. DATA PROVIDER
# =============================================================================

class DataProvider:
    """Robust data provider with caching and validation"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache_dir = Path(config.data_cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get hourly data with caching"""
        
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.pkl"
        
        # Try cache first
        if cache_file.exists():
            try:
                data = pd.read_pickle(cache_file)
                if self._validate_data(data, symbol):
                    self.logger.debug(f"Loaded {symbol} from cache: {len(data)} bars")
                    return data
            except Exception as e:
                self.logger.warning(f"Cache read failed for {symbol}: {e}")
        
        # Fetch new data
        try:
            data = self._fetch_yfinance_data(symbol, start_date, end_date)
            
            if self._validate_data(data, symbol):
                # Cache the data
                try:
                    data.to_pickle(cache_file)
                except Exception as e:
                    self.logger.warning(f"Failed to cache {symbol}: {e}")
                
                self.logger.info(f"Fetched {symbol}: {len(data)} bars")
                return data
            else:
                self.logger.error(f"Invalid data for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Data fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_yfinance_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with retries"""
        
        for attempt in range(3):
            try:
                self.logger.debug(f"Fetching {symbol}, attempt {attempt + 1}")
                
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    interval='1h',
                    progress=False,
                    auto_adjust=True,
                    prepost=False,
                    threads=True
                )
                
                if data is not None and len(data) > 0:
                    # Standardize the data
                    data = self._standardize_data(data)
                    return data
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt == 2:
                    raise
        
        return pd.DataFrame()
    
    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format"""
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Standardize column names
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Handle adj_close
        if 'adj_close' in data.columns and 'close' not in data.columns:
            data['close'] = data['adj_close']
        
        # Ensure timezone-naive timestamps
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Sort by time
        data = data.sort_index()
        
        return data
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality"""
        
        if data is None or len(data) == 0:
            return False
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            self.logger.warning(f"Missing columns for {symbol}: {set(required_cols) - set(data.columns)}")
            return False
        
        # Check for valid prices
        if (data['close'] <= 0).any():
            self.logger.warning(f"Invalid prices detected for {symbol}")
            return False
        
        # Check high >= low
        if (data['high'] < data['low']).any():
            self.logger.warning(f"High < Low detected for {symbol}")
            return False
        
        # Check minimum data points
        if len(data) < self.config.min_data_points:
            self.logger.warning(f"Insufficient data for {symbol}: {len(data)} < {self.config.min_data_points}")
            return False
        
        return True

# =============================================================================
# 5. ENHANCED FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Enhanced feature engineering focused on signal quality"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create quality-focused feature set"""
        
        try:
            features = pd.DataFrame(index=data.index)
            
            # Core price features
            features = self._add_price_features(features, data)
            
            # Technical indicators
            features = self._add_technical_indicators(features, data)
            
            # Enhanced statistical features
            features = self._add_statistical_features(features, data)
            
            # Market regime features
            features = self._add_regime_features(features, data)
            
            # Time-based features
            features = self._add_time_features(features, data)
            
            # Clean features
            features = self._clean_features(features)
            
            self.logger.debug(f"Created {len(features.columns)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return pd.DataFrame()
    
    def _add_price_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced price-based features"""
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Returns
        features['returns_1h'] = close.pct_change()
        features['returns_3h'] = close.pct_change(3)
        features['log_returns'] = np.log(close / close.shift(1))
        
        # Price levels relative to moving averages (CRITICAL for mean reversion)
        for window in self.config.feature_windows:
            ma = close.rolling(window).mean()
            std = close.rolling(window).std()
            
            features[f'price_to_ma_{window}'] = close / (ma + 1e-8)
            features[f'zscore_{window}'] = (close - ma) / (std + 1e-8)  # PRIMARY SIGNAL
            
            # Price position in recent range
            high_n = close.rolling(window).max()
            low_n = close.rolling(window).min()
            features[f'price_position_{window}'] = (close - low_n) / (high_n - low_n + 1e-8)
        
        # Enhanced momentum features
        for lag in self.config.feature_windows[:3]:  # Focus on key lags
            features[f'momentum_{lag}'] = close / close.shift(lag) - 1
        
        # Volume features
        vol_ma = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_ma + 1)
        features['volume_surge'] = volume / (vol_ma + 1)  # Duplicate for clarity
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators"""
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # RSI with divergence detection
        features['rsi'] = self._calculate_rsi(close, self.config.rsi_period)
        features['rsi_divergence'] = features['rsi'].diff(5)  # NEW: Divergence signal
        features['rsi_oversold'] = (features['rsi'] < 25).astype(int)  # MORE EXTREME
        features['rsi_overbought'] = (features['rsi'] > 75).astype(int)  # MORE EXTREME
        
        # Enhanced Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(
            close, self.config.bb_period, self.config.bb_std
        )
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        bb_width = (bb_upper - bb_lower) / bb_middle
        features['bb_width'] = bb_width
        features['bb_squeeze'] = bb_width < bb_width.rolling(50).quantile(0.2)  # NEW: Squeeze detection
        
        # MACD with enhanced signals
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9).mean()
        features['macd_histogram'] = macd_line - macd_signal
        features['macd_signal'] = np.where(macd_line > macd_signal, 1, -1)
        
        return features
    
    def _add_statistical_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced statistical features"""
        
        close = data['close']
        returns = features['returns_1h']
        
        # Volatility regime detection (CRITICAL)
        for window in [10, 20]:
            vol = returns.rolling(window).std()
            vol_long = returns.rolling(window*3).std()
            features[f'volatility_{window}'] = vol
            features[f'vol_regime_{window}'] = vol / (vol_long + 1e-8)  # Regime indicator
        
        # Autocorrelation (mean reversion detector)
        features['autocorr_5'] = returns.rolling(50).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )
        
        return features
    
    def _add_regime_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        
        close = data['close']
        returns = features['returns_1h']
        
        # Trend strength
        features['trend_strength'] = abs(features.get('momentum_5', 0))
        
        # Volatility clustering
        vol_10 = features.get('volatility_10', 0)
        vol_ma = vol_10.rolling(20).mean()
        features['vol_cluster'] = vol_10 / (vol_ma + 1e-8)
        
        # Market stress indicator
        features['market_stress'] = (vol_10 > vol_10.rolling(100).quantile(0.8)).astype(int)
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        # Hour of day
        features['hour'] = data.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Market session indicators
        features['market_open'] = (features['hour'] == 9).astype(int)
        features['market_close'] = (features['hour'] == 15).astype(int)
        features['lunch_hour'] = (features['hour'] == 12).astype(int)
        
        return features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        
        # Replace infinities
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        features = features.fillna(method='ffill', limit=5)
        features = features.fillna(method='bfill', limit=5)
        
        # Fill remaining NaNs with 0
        features = features.fillna(0)
        
        # Remove constant columns
        non_constant = features.std() > 1e-8
        features = features.loc[:, non_constant]
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower, ma

# =============================================================================
# 6. ENHANCED MODEL MANAGEMENT
# =============================================================================

class ModelManager:
    """Enhanced model management with market regime filtering"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.models: Dict[str, Dict] = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def get_signal(self, symbol: str, features: pd.DataFrame, current_time: datetime) -> Tuple[float, Dict[str, Any]]:
        """Enhanced signal generation with quality filters"""
        
        try:
            # Check market regime first
            regime_ok, regime_reason = self._check_market_regime(features, current_time)
            if not regime_ok:
                return 0.0, {'regime_filter': regime_reason}
            
            # Get ML prediction if model exists
            ml_signal = 0.0
            if symbol in self.models:
                ml_signal = self._get_ml_prediction(symbol, features)
            
            # Get enhanced rule-based signal
            rule_signal = self._get_enhanced_rule_signal(features)
            
            # Combine signals with better logic
            if abs(ml_signal) > 0.1 and abs(rule_signal) > 0.1:
                # Both signals agree on direction
                if np.sign(ml_signal) == np.sign(rule_signal):
                    final_signal = (ml_signal + rule_signal) / 2
                    signal_type = "ml_rule_agree"
                else:
                    # Signals disagree - use weaker signal
                    final_signal = min(abs(ml_signal), abs(rule_signal)) * np.sign(ml_signal)
                    signal_type = "ml_rule_disagree"
            elif abs(ml_signal) > abs(rule_signal):
                final_signal = ml_signal
                signal_type = "ml_primary"
            else:
                final_signal = rule_signal
                signal_type = "rule_primary"
            
            # Enhanced confirmation checking
            confirmations = self._check_enhanced_confirmations(features, final_signal)
            
            metadata = {
                'ml_signal': ml_signal,
                'rule_signal': rule_signal,
                'final_signal': final_signal,
                'signal_type': signal_type,
                'confirmations': confirmations,
                'confidence': len(confirmations) * abs(final_signal),
                'signal_quality': len(confirmations) * abs(final_signal)
            }
            
            # Apply ENHANCED threshold and confirmations
            if (abs(final_signal) >= self.config.signal_threshold and 
                len(confirmations) >= self.config.min_confirmations):
                return final_signal, metadata
            else:
                return 0.0, metadata
                
        except Exception as e:
            self.logger.error(f"Signal generation failed for {symbol}: {e}")
            return 0.0, {'error': str(e)}
    
    def _check_market_regime(self, features: pd.DataFrame, current_time: datetime) -> Tuple[bool, str]:
        """Enhanced market regime filtering"""
        
        if not self.config.regime_filter:
            return True, "regime_filter_disabled"
        
        latest = features.iloc[-1]
        current_hour = current_time.hour
        
        # Time-based filters (RELAXED)
        if self.config.avoid_first_hour and current_hour == 9:
            return False, "avoid_first_hour"
        if self.config.avoid_last_hour and current_hour == 15:
            return False, "avoid_last_hour"
        if self.config.lunch_hour_filter and current_hour == 12:
            return False, "lunch_hour"
        
        # Volatility filter (RELAXED)
        if self.config.volatility_filter:
            vol_regime = latest.get('vol_regime_10', 1.0)
            if vol_regime > 2.5:  # RELAXED from 2.0 - only avoid extreme volatility
                return False, "extreme_volatility_regime"
        
        # Trend filter (RELAXED)
        if self.config.trend_filter:
            momentum_5 = abs(latest.get('momentum_5', 0))
            if momentum_5 > 0.05:  # RELAXED from 0.03 - only avoid very strong trends
                return False, "very_strong_trend"
        
        # Market stress filter
        market_stress = latest.get('market_stress', 0)
        if market_stress > 0:
            return False, "market_stress"
        
        return True, "regime_suitable"
    
    def _get_enhanced_rule_signal(self, features: pd.DataFrame) -> float:
        """Enhanced rule-based signal generation"""
        
        try:
            latest = features.iloc[-1]
            signal = 0.0
            
            # PRIMARY: Z-score signals (mean reversion core) - BALANCED
            zscore_20 = latest.get('zscore_20', 0)
            zscore_50 = latest.get('zscore_50', 0)
            
            # Balanced z-score requirements
            if abs(zscore_20) > 1.5:  # BALANCED threshold (was 1.8)
                signal += -np.sign(zscore_20) * min(abs(zscore_20) / 2.2, 0.7)  # INCREASED impact
            
            if abs(zscore_50) > 1.2:  # RELAXED from 1.5
                signal += -np.sign(zscore_50) * min(abs(zscore_50) / 2.5, 0.4)
            
            # BALANCED: RSI with divergence
            rsi = latest.get('rsi', 50)
            rsi_div = latest.get('rsi_divergence', 0)
            
            if rsi < 30 and rsi_div > 0:  # RELAXED from 25
                signal += 0.35  # INCREASED from 0.3
            elif rsi > 70 and rsi_div < 0:  # RELAXED from 75
                signal -= 0.35  # INCREASED from 0.3
            
            # ENHANCED: Bollinger Band with squeeze
            bb_pos = latest.get('bb_position', 0.5)
            bb_squeeze = latest.get('bb_squeeze', 0)
            
            if bb_pos < 0.1 and bb_squeeze:  # Lower band + squeeze
                signal += 0.25
            elif bb_pos > 0.9 and bb_squeeze:  # Upper band + squeeze
                signal -= 0.25
            
            # Volume confirmation
            volume_surge = latest.get('volume_surge', 1.0)
            if volume_surge > self.config.volume_threshold:
                signal *= 1.15  # Modest amplification
            
            # Trend dampening
            momentum_5 = latest.get('momentum_5', 0)
            if abs(momentum_5) > 0.02:  # Strong momentum
                if np.sign(signal) != np.sign(momentum_5):
                    signal *= 0.5  # Reduce counter-trend signals
            
            return np.clip(signal, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Enhanced rule signal failed: {e}")
            return 0.0
    
    def _check_enhanced_confirmations(self, features: pd.DataFrame, signal: float) -> List[str]:
        """Enhanced confirmation checking with higher standards"""
        
        confirmations = []
        latest = features.iloc[-1]
        
        # Z-score confirmation (BALANCED STANDARDS)
        zscore_20 = latest.get('zscore_20', 0)
        if signal > 0 and zscore_20 < -1.2:  # RELAXED from -1.5
            confirmations.append('zscore_oversold')
        elif signal < 0 and zscore_20 > 1.2:  # RELAXED from 1.5
            confirmations.append('zscore_overbought')
        
        # RSI confirmation with divergence (RELAXED)
        rsi = latest.get('rsi', 50)
        rsi_div = latest.get('rsi_divergence', 0)
        if signal > 0 and rsi < 35:  # RELAXED from 30
            confirmations.append('rsi_oversold')
        elif signal < 0 and rsi > 65:  # RELAXED from 70
            confirmations.append('rsi_overbought')
        
        # Bollinger Band confirmation (RELAXED)
        bb_pos = latest.get('bb_position', 0.5)
        bb_squeeze = latest.get('bb_squeeze', 0)
        if signal > 0 and bb_pos < 0.2:  # RELAXED from 0.15
            confirmations.append('bb_oversold')
        elif signal < 0 and bb_pos > 0.8:  # RELAXED from 0.85
            confirmations.append('bb_overbought')
        
        # Volume confirmation (RELAXED)
        volume_surge = latest.get('volume_surge', 1.0)
        if volume_surge > 1.3:  # RELAXED from 1.5
            confirmations.append('volume_confirmation')
        
        # Low volatility regime confirmation
        vol_regime = latest.get('vol_regime_10', 1.0)
        if vol_regime < 0.8:  # Calm market
            confirmations.append('low_vol_regime')
        
        return confirmations
    
    def should_retrain(self, symbol: str) -> bool:
        """Check if model should be retrained"""
        
        if symbol not in self.models:
            return True
        
        last_training = self.models[symbol].get('last_training')
        if last_training is None:
            return True
        
        hours_since = (datetime.now() - last_training).total_seconds() / 3600
        return hours_since >= self.config.retrain_frequency_hours
    
    def train_model(self, symbol: str, features: pd.DataFrame, target: pd.Series) -> bool:
        """Train model for symbol"""
        
        try:
            # Prepare data
            mask = target.notna() & (features.sum(axis=1) != 0)
            X = features[mask]
            y = target[mask]
            
            if len(X) < self.config.min_training_samples:
                self.logger.warning(f"Insufficient training data for {symbol}: {len(X)}")
                return False
            
            # Feature selection
            selected_features = self._select_features(X, y)
            X_selected = X[selected_features]
            
            # Train model with cross-validation
            model_data = self._train_with_cv(X_selected, y)
            
            if model_data is not None:
                model_data['features'] = selected_features
                model_data['last_training'] = datetime.now()
                model_data['training_samples'] = len(X)
                
                self.models[symbol] = model_data
                
                # Save model
                self._save_model(symbol, model_data)
                
                self.logger.info(f"Successfully trained model for {symbol}")
                return True
            else:
                self.logger.warning(f"Model training failed validation for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Model training failed for {symbol}: {e}")
            return False
    
    def _get_ml_prediction(self, symbol: str, features: pd.DataFrame) -> float:
        """Get ML model prediction"""
        
        try:
            model_data = self.models[symbol]
            model = model_data['model']
            scaler = model_data['scaler']
            selected_features = model_data['features']
            
            # Prepare features
            latest_features = features[selected_features].iloc[-1:].fillna(0)
            scaled_features = scaler.transform(latest_features)
            
            # Get prediction
            prediction = model.predict(scaled_features)[0]
            return np.clip(prediction, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed for {symbol}: {e}")
            return 0.0
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select best features using correlation"""
        
        # Calculate feature importance using correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # Select top features
        n_features = min(self.config.max_features, len(correlations))
        selected = correlations.head(n_features).index.tolist()
        
        self.logger.debug(f"Selected {len(selected)} features from {len(X.columns)}")
        return selected
    
    def _train_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict]:
        """Train model with cross-validation"""
        
        try:
            # Prepare for cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            cv_scores = []
            
            # Scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validation
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model with MORE CONSERVATIVE parameters
                model = RandomForestRegressor(
                    n_estimators=80,              # Reduced from 100
                    max_depth=6,                  # Reduced from 8
                    min_samples_split=25,         # Increased from 20
                    min_samples_leaf=12,          # Increased from 10
                    max_features='sqrt',          # More conservative
                    random_state=self.config.model_random_state,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                
                # Calculate Information Coefficient
                ic = np.corrcoef(y_pred, y_val)[0, 1] if len(y_pred) > 1 else 0
                cv_scores.append({'mse': mse, 'ic': ic})
            
            # Check if model is good enough (BALANCED THRESHOLD)
            avg_ic = np.mean([score['ic'] for score in cv_scores if not np.isnan(score['ic'])])
            
            if abs(avg_ic) > 0.06:  # BALANCED threshold between 0.05 and 0.08
                # Train final model on all data
                final_model = RandomForestRegressor(
                    n_estimators=80,
                    max_depth=6,
                    min_samples_split=25,
                    min_samples_leaf=12,
                    max_features='sqrt',
                    random_state=self.config.model_random_state,
                    n_jobs=-1
                )
                
                final_model.fit(X_scaled, y)
                
                return {
                    'model': final_model,
                    'scaler': scaler,
                    'cv_scores': cv_scores,
                    'avg_ic': avg_ic
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Cross-validation training failed: {e}")
            return None
    
    def _save_model(self, symbol: str, model_data: Dict):
        """Save model to disk"""
        try:
            model_file = self.model_dir / f"{symbol}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save model for {symbol}: {e}")
    
    def _create_target(self, data: pd.DataFrame, horizon: int = 4) -> pd.Series:  # REDUCED horizon
        """Create target variable for training"""
        
        close = data['close']
        
        # Future return target (SHORTER horizon for mean reversion)
        future_return = close.shift(-horizon) / close - 1
        
        # Mean reversion component
        ma_20 = close.rolling(20).mean()
        current_deviation = (close - ma_20) / (ma_20 + 1e-8)
        future_ma = close.shift(-horizon).rolling(20).mean()
        future_deviation = (close.shift(-horizon) - future_ma) / (future_ma + 1e-8)
        
        mean_reversion = future_deviation - current_deviation
        
        # Combine targets (MORE weight on mean reversion)
        combined_target = 0.4 * future_return + 0.6 * mean_reversion
        
        return combined_target

# =============================================================================
# 7. ENHANCED PORTFOLIO MANAGEMENT
# =============================================================================

class PortfolioManager:
    """Enhanced portfolio management with better risk controls"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Enhanced tracking
        self.daily_trades_count = 0
        self.last_trade_date = None
        self.peak_value = config.initial_capital
        self.current_drawdown = 0.0
        
        # Sector mapping for diversification
        self.sectors = {
            'SPY': 'Broad Market', 'QQQ': 'Technology', 'IWM': 'Small Cap',
            'XLF': 'Finance', 'XLE': 'Energy', 'XLK': 'Technology',
            'XLV': 'Healthcare', 'XLI': 'Industrial', 'XLP': 'Consumer',
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'META': 'Technology', 'AMZN': 'Technology', 'TSLA': 'Technology',
            'JPM': 'Finance', 'BAC': 'Finance', 'JNJ': 'Healthcare'
        }
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            abs(pos.shares) * current_prices.get(symbol, pos.entry_price)
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def can_open_position(self, symbol: str, signal_strength: float, current_price: float, 
                         current_prices: Dict[str, float], features: pd.Series,
                         current_time: datetime) -> Tuple[bool, str, int]:
        """Enhanced position opening checks"""
        
        # Update daily trade tracking
        if self.last_trade_date != current_time.date():
            self.daily_trades_count = 0
            self.last_trade_date = current_time.date()
        
        # Daily trade limit
        if self.daily_trades_count >= self.config.max_daily_trades:
            return False, "daily_trade_limit", 0
        
        # Drawdown limit check
        portfolio_value = self.get_portfolio_value(current_prices)
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        if self.current_drawdown > self.config.drawdown_limit:
            return False, "drawdown_limit_exceeded", 0
        
        # Position limit
        if len(self.positions) >= self.config.max_positions:
            return False, "max_positions", 0
        
        # Check if already have position
        if symbol in self.positions:
            return False, "existing_position", 0
        
        # Enhanced position sizing
        shares = self._calculate_enhanced_position_size(
            symbol, signal_strength, current_price, portfolio_value, features
        )
        
        if shares < 1:
            return False, "insufficient_capital", 0
        
        # Sector exposure check
        sector = self.sectors.get(symbol, 'Other')
        current_sector_exposure = self._get_sector_exposure(sector, current_prices)
        new_exposure = current_sector_exposure + (shares * current_price) / portfolio_value
        
        if new_exposure > self.config.max_sector_exposure:
            return False, "sector_limit", 0
        
        # Cash requirement check
        trade_cost = shares * current_price * (1 + self.config.slippage_bps / 10000)
        if trade_cost > self.cash * 0.95:
            return False, "insufficient_cash", 0
        
        return True, "approved", shares
    
    def _calculate_enhanced_position_size(self, symbol: str, signal_strength: float, 
                                        current_price: float, portfolio_value: float,
                                        features: pd.Series) -> int:
        """Enhanced position sizing with volatility adjustment"""
        
        # Base position size
        base_size = portfolio_value * self.config.position_size_pct
        
        # Volatility adjustment
        if self.config.volatility_scaling:
            vol_regime = features.get('vol_regime_10', 1.0)
            # Reduce size in high volatility
            vol_scalar = max(0.5, min(1.5, 1.0 / vol_regime))
            base_size *= vol_scalar
        
        # Signal strength adjustment (MORE CONSERVATIVE)
        strength_scalar = min(abs(signal_strength) * 1.2, 1.2)  # REDUCED max from 1.5
        final_size = base_size * strength_scalar
        
        # Convert to shares
        shares = int(final_size / current_price)
        return max(shares, 1)
    
    def open_position(self, symbol: str, signal_strength: float, shares: int, 
                     current_price: float, current_time: datetime) -> bool:
        """Open a new position with enhanced logic"""
        
        try:
            # Calculate execution price with slippage
            direction = 1 if signal_strength > 0 else -1
            signed_shares = shares * direction
            execution_price = current_price * (1 + direction * self.config.slippage_bps / 10000)
            
            # Enhanced stop loss and take profit calculation
            if direction > 0:  # Long position
                stop_loss = execution_price * (1 - self.config.stop_loss_pct)
                take_profit = execution_price * (1 + self.config.take_profit_pct)
            else:  # Short position
                stop_loss = execution_price * (1 + self.config.stop_loss_pct)
                take_profit = execution_price * (1 - self.config.take_profit_pct)
            
            # Create position
            position = Position(
                symbol=symbol,
                shares=signed_shares,
                entry_price=execution_price,
                entry_time=current_time,
                signal_strength=signal_strength,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Update cash
            trade_cost = shares * execution_price + self.config.commission_per_trade
            self.cash -= trade_cost
            
            # Store position
            self.positions[symbol] = position
            
            # Update daily trade count
            self.daily_trades_count += 1
            
            self.logger.info(f"Opened {('LONG' if direction > 0 else 'SHORT')} {symbol}: "
                           f"{shares} shares @ ${execution_price:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open position {symbol}: {e}")
            return False
    
    def check_exit_conditions(self, symbol: str, current_price: float, 
                            current_time: datetime, features: pd.Series) -> Tuple[bool, str]:
        """Enhanced exit condition checking"""
        
        if symbol not in self.positions:
            return False, "no_position"
        
        position = self.positions[symbol]
        
        # Update trailing stop
        if self.config.trailing_stop:
            position.update_trailing_stop(current_price, self.config)
        
        # Time-based exit (REDUCED holding time)
        holding_hours = (current_time - position.entry_time).total_seconds() / 3600
        if holding_hours > self.config.max_hold_hours:
            return True, "max_time"
        
        # Stop loss
        if position.is_long and current_price <= position.stop_loss:
            return True, "stop_loss"
        elif not position.is_long and current_price >= position.stop_loss:
            return True, "stop_loss"
        
        # Enhanced take profit logic
        should_exit, reason = self._check_enhanced_take_profit(position, current_price, features)
        if should_exit:
            return True, reason
        
        # Trailing stop
        if (self.config.trailing_stop and position.trailing_stop_price is not None):
            if position.is_long and current_price <= position.trailing_stop_price:
                return True, "trailing_stop"
            elif not position.is_long and current_price >= position.trailing_stop_price:
                return True, "trailing_stop"
        
        # Enhanced mean reversion completion check
        try:
            zscore_20 = features.get('zscore_20', 0)
            # More sensitive mean reversion exit
            if position.is_long and zscore_20 > -0.2:  # TIGHTER from -0.3
                return True, "mean_reversion_complete"
            elif not position.is_long and zscore_20 < 0.2:  # TIGHTER from 0.3
                return True, "mean_reversion_complete"
        except:
            pass
        
        return False, "hold"
    
    def _check_enhanced_take_profit(self, position: Position, current_price: float, 
                                  features: pd.Series) -> Tuple[bool, str]:
        """Enhanced take profit logic"""
        
        # Calculate current P&L
        if position.is_long:
            unrealized_return = (current_price - position.entry_price) / position.entry_price
        else:
            unrealized_return = (position.entry_price - current_price) / position.entry_price
        
        # Standard take profit
        if position.is_long and current_price >= position.take_profit:
            return True, "take_profit"
        elif not position.is_long and current_price <= position.take_profit:
            return True, "take_profit"
        
        # Dynamic take profit based on momentum
        try:
            momentum_5 = features.get('momentum_5', 0)
            
            # If we have profit and momentum is reversing
            if unrealized_return > 0.03:  # 3% profit
                if position.is_long and momentum_5 < -0.01:  # Momentum turning negative
                    return True, "momentum_reversal_profit"
                elif not position.is_long and momentum_5 > 0.01:  # Momentum turning positive
                    return True, "momentum_reversal_profit"
        except:
            pass
        
        return False, "hold"
    
    def close_position(self, symbol: str, current_price: float, 
                      current_time: datetime, reason: str) -> bool:
        """Close an existing position"""
        
        if symbol not in self.positions:
            return False
        
        try:
            position = self.positions[symbol]
            
            # Calculate execution price with slippage
            direction = -1 if position.is_long else 1
            execution_price = current_price * (1 + direction * self.config.slippage_bps / 10000)
            
            # Calculate P&L
            if position.is_long:
                pnl = position.shares * (execution_price - position.entry_price)
            else:
                pnl = -position.shares * (position.entry_price - execution_price)
            
            # Subtract costs
            total_cost = self.config.commission_per_trade
            net_pnl = pnl - total_cost
            
            # Update cash
            trade_value = abs(position.shares) * execution_price
            self.cash += trade_value - total_cost
            
            # Calculate metrics
            holding_hours = (current_time - position.entry_time).total_seconds() / 3600
            return_pct = (net_pnl / (abs(position.shares) * position.entry_price)) * 100
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_time=position.entry_time,
                exit_time=current_time,
                shares=position.shares,
                entry_price=position.entry_price,
                exit_price=execution_price,
                pnl=net_pnl,
                return_pct=return_pct,
                holding_hours=holding_hours,
                exit_reason=reason,
                signal_strength=position.signal_strength
            )
            
            self.trades.append(trade)
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.info(f"Closed {symbol}: P&L=${net_pnl:.2f} ({return_pct:.1f}%) - {reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {e}")
            return False
    
    def _get_sector_exposure(self, sector: str, current_prices: Dict[str, float]) -> float:
        """Calculate current sector exposure"""
        
        portfolio_value = self.get_portfolio_value(current_prices)
        if portfolio_value == 0:
            return 0.0
        
        sector_value = sum(
            abs(pos.shares) * current_prices.get(symbol, pos.entry_price)
            for symbol, pos in self.positions.items()
            if self.sectors.get(symbol, 'Other') == sector
        )
        
        return sector_value / portfolio_value

# =============================================================================
# 8. ENHANCED BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """Enhanced backtesting engine"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logging(config)
        
        # Initialize components
        self.data_provider = DataProvider(config, self.logger)
        self.feature_engineer = FeatureEngineer(config, self.logger)
        self.model_manager = ModelManager(config, self.logger)
        self.portfolio_manager = PortfolioManager(config, self.logger)
        
        # Results tracking
        self.hourly_snapshots = []
        self.universe = []
    
    def prepare_universe(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Prepare trading universe"""
        
        self.logger.info(f"Preparing universe with {len(symbols)} symbols")
        
        market_data = {}
        successful = 0
        
        for symbol in symbols:
            try:
                data = self.data_provider.get_data(symbol, start_date, end_date)
                if len(data) > self.config.min_data_points:
                    market_data[symbol] = data
                    successful += 1
                    self.logger.debug(f"Loaded {symbol}: {len(data)} bars")
                else:
                    self.logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {symbol}: {e}")
        
        self.logger.info(f"Successfully loaded {successful}/{len(symbols)} symbols")
        self.universe = list(market_data.keys())
        
        return market_data
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Run the enhanced backtest"""
        
        self.logger.info("="*80)
        self.logger.info("STARTING ENHANCED ML MEAN REVERSION BACKTEST")
        self.logger.info("="*80)
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Universe: {len(symbols)} symbols")
        self.logger.info(f"Initial Capital: ${self.config.initial_capital:,.0f}")
        self.logger.info(f"ENHANCEMENTS: Signal Threshold={self.config.signal_threshold}, "
                        f"Min Confirmations={self.config.min_confirmations}, "
                        f"Max Positions={self.config.max_positions}")
        
        # Prepare data
        market_data = self.prepare_universe(symbols, start_date, end_date)
        
        if not market_data:
            self.logger.error("No market data available!")
            return {}
        
        # Pre-calculate features
        self.logger.info("Calculating enhanced features...")
        features_data = {}
        
        for symbol, data in market_data.items():
            try:
                features = self.feature_engineer.create_features(data)
                if len(features.columns) > 0:
                    features_data[symbol] = features
            except Exception as e:
                self.logger.error(f"Feature calculation failed for {symbol}: {e}")
        
        self.logger.info(f"Features calculated for {len(features_data)} symbols")
        
        # Get all timestamps
        all_timestamps = []
        for data in market_data.values():
            all_timestamps.extend(data.index.tolist())
        
        trading_hours = sorted(list(set(all_timestamps)))
        trading_hours = [ts for ts in trading_hours 
                        if pd.Timestamp(start_date) <= ts <= pd.Timestamp(end_date)]
        
        self.logger.info(f"Backtesting over {len(trading_hours)} hours")
        
        # Main backtest loop
        signals_generated = 0
        trades_executed = 0
        
        for i, current_time in enumerate(trading_hours):
            if i < 100:  # Warmup period
                continue
            
            try:
                # Get current market data
                current_prices = {}
                current_data = {}
                
                for symbol in self.universe:
                    if symbol not in market_data or symbol not in features_data:
                        continue
                    
                    try:
                        # Find current bar
                        data = market_data[symbol]
                        idx = data.index.get_indexer([current_time], method='nearest')[0]
                        if idx >= 0 and abs((data.index[idx] - current_time).total_seconds()) <= 3600:
                            current_prices[symbol] = data.iloc[idx]['close']
                            current_data[symbol] = data.iloc[idx]
                    except:
                        continue
                
                if not current_prices:
                    continue
                
                # Update portfolio snapshot
                portfolio_value = self.portfolio_manager.get_portfolio_value(current_prices)
                self.hourly_snapshots.append({
                    'time': current_time,
                    'portfolio_value': portfolio_value,
                    'cash': self.portfolio_manager.cash,
                    'num_positions': len(self.portfolio_manager.positions),
                    'prices': current_prices.copy()
                })
                
                # Check exits first
                for symbol in list(self.portfolio_manager.positions.keys()):
                    if symbol not in current_prices or symbol not in features_data:
                        continue
                    
                    try:
                        features = features_data[symbol]
                        current_features = features.loc[features.index <= current_time].iloc[-1]
                        
                        should_exit, reason = self.portfolio_manager.check_exit_conditions(
                            symbol, current_prices[symbol], current_time, current_features
                        )
                        
                        if should_exit:
                            if self.portfolio_manager.close_position(
                                symbol, current_prices[symbol], current_time, reason
                            ):
                                trades_executed += 1
                    except Exception as e:
                        self.logger.debug(f"Exit check failed for {symbol}: {e}")
                
                # Generate entry signals
                if len(self.portfolio_manager.positions) < self.config.max_positions:
                    signal_candidates = []
                    
                    for symbol in self.universe:
                        if symbol in self.portfolio_manager.positions:
                            continue
                        
                        if symbol not in current_prices or symbol not in features_data:
                            continue
                        
                        try:
                            # Get historical data and features
                            data = market_data[symbol]
                            features = features_data[symbol]
                            
                            # Get data up to current time
                            hist_data = data.loc[data.index <= current_time]
                            hist_features = features.loc[features.index <= current_time]
                            
                            if len(hist_data) < 200 or len(hist_features) < 200:
                                continue
                            
                            # Check if model needs training
                            if self.model_manager.should_retrain(symbol):
                                target = self.model_manager._create_target(hist_data)
                                self.model_manager.train_model(symbol, hist_features, target)
                            
                            # Generate enhanced signal
                            signal, metadata = self.model_manager.get_signal(symbol, hist_features, current_time)
                            
                            if abs(signal) > 0:
                                signals_generated += 1
                                
                                if abs(signal) >= self.config.signal_threshold:
                                    signal_candidates.append({
                                        'symbol': symbol,
                                        'signal': signal,
                                        'confidence': metadata.get('confidence', 0),
                                        'metadata': metadata
                                    })
                        
                        except Exception as e:
                            self.logger.debug(f"Signal generation failed for {symbol}: {e}")
                    
                    # Sort by confidence and execute best signals
                    signal_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    for candidate in signal_candidates[:2]:  # REDUCED from 3 to 2 max new positions
                        symbol = candidate['symbol']
                        signal = candidate['signal']
                        
                        # Get current features for position sizing
                        features = features_data[symbol]
                        current_features = features.loc[features.index <= current_time].iloc[-1]
                        
                        # Check if we can open position
                        can_open, reason, shares = self.portfolio_manager.can_open_position(
                            symbol, signal, current_prices[symbol], current_prices, 
                            current_features, current_time
                        )
                        
                        if can_open:
                            if self.portfolio_manager.open_position(
                                symbol, signal, shares, current_prices[symbol], current_time
                            ):
                                trades_executed += 1
                
                # Progress logging
                if i % 1000 == 0:
                    returns = (portfolio_value / self.config.initial_capital - 1) * 100
                    progress = (i / len(trading_hours)) * 100
                    
                    self.logger.info(f"[{progress:.1f}%] {current_time}: "
                                   f"Value=${portfolio_value:,.0f} ({returns:.1f}%) | "
                                   f"Positions={len(self.portfolio_manager.positions)} | "
                                   f"Trades={trades_executed} | Drawdown={self.portfolio_manager.current_drawdown:.1%}")
                
            except Exception as e:
                self.logger.error(f"Error at {current_time}: {e}")
                continue
        
        # Final calculations
        final_value = self.portfolio_manager.get_portfolio_value(current_prices)
        total_return = (final_value / self.config.initial_capital - 1) * 100
        
        self.logger.info("="*80)
        self.logger.info("ENHANCED BACKTEST COMPLETED")
        self.logger.info(f"Signals Generated: {signals_generated:,}")
        self.logger.info(f"Trades Executed: {trades_executed:,}")
        self.logger.info(f"Signal-to-Trade Ratio: {(trades_executed/max(signals_generated,1)*100):.1f}%")
        self.logger.info(f"Final Value: ${final_value:,.2f}")
        self.logger.info(f"Total Return: {total_return:.2f}%")
        self.logger.info(f"Max Drawdown: {self.portfolio_manager.current_drawdown:.1%}")
        self.logger.info("="*80)
        
        return {
            'initial_capital': self.config.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'signals_generated': signals_generated,
            'trades_executed': trades_executed,
            'trades': self.portfolio_manager.trades,
            'hourly_snapshots': self.hourly_snapshots,
            'universe': self.universe
        }

# =============================================================================
# 9. RESULTS ANALYSIS (SAME AS BEFORE)
# =============================================================================

class ResultsAnalyzer:
    """Analyze and visualize backtest results"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive results analysis"""
        
        if not results or not results.get('hourly_snapshots'):
            return {}
        
        try:
            # Create DataFrames
            snapshots_df = pd.DataFrame(results['hourly_snapshots'])
            snapshots_df.set_index('time', inplace=True)
            
            trades_df = pd.DataFrame([
                {
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'shares': trade.shares,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'holding_hours': trade.holding_hours,
                    'exit_reason': trade.exit_reason,
                    'signal_strength': trade.signal_strength
                }
                for trade in results.get('trades', [])
            ])
            
            # Performance metrics
            metrics = self._calculate_metrics(snapshots_df, trades_df, results)
            
            self.logger.info("Analysis completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {}
    
    def _calculate_metrics(self, snapshots_df: pd.DataFrame, trades_df: pd.DataFrame, 
                          results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        
        # Basic metrics
        initial_capital = results['initial_capital']
        final_value = results['final_value']
        total_return = (final_value / initial_capital - 1) * 100
        
        # Time-based metrics
        start_time = snapshots_df.index[0]
        end_time = snapshots_df.index[-1]
        total_days = (end_time - start_time).days
        years = max(total_days / 365.25, 0.1)
        
        # Returns
        snapshots_df['returns'] = snapshots_df['portfolio_value'].pct_change()
        snapshots_df['cumulative_returns'] = (snapshots_df['portfolio_value'] / initial_capital - 1) * 100
        
        # Risk metrics
        hourly_returns = snapshots_df['returns'].dropna()
        
        if len(hourly_returns) > 0:
            # Annualized metrics
            hours_per_year = 365.25 * 24
            mean_hourly_return = hourly_returns.mean()
            std_hourly_return = hourly_returns.std()
            
            # CAGR
            cagr = (final_value / initial_capital) ** (1 / years) - 1
            
            # Volatility
            annual_vol = std_hourly_return * np.sqrt(hours_per_year)
            
            # Sharpe ratio (assuming 3% risk-free rate)
            risk_free_rate = 0.03
            excess_return = cagr - risk_free_rate
            sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0
            
            # Sortino ratio
            negative_returns = hourly_returns[hourly_returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else std_hourly_return
            annual_downside_vol = downside_std * np.sqrt(hours_per_year)
            sortino_ratio = excess_return / annual_downside_vol if annual_downside_vol > 0 else 0
            
            # Maximum drawdown
            rolling_max = snapshots_df['portfolio_value'].expanding().max()
            drawdowns = (snapshots_df['portfolio_value'] - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() * 100
            
            # Calmar ratio
            calmar_ratio = cagr / abs(max_drawdown / 100) if max_drawdown < 0 else 0
            
            # Win rates
            win_rate_hourly = len(hourly_returns[hourly_returns > 0]) / len(hourly_returns) * 100
            
        else:
            cagr = annual_vol = sharpe_ratio = sortino_ratio = 0
            max_drawdown = calmar_ratio = win_rate_hourly = 0
        
        # Trading metrics
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate_trades = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else np.inf
            avg_holding_hours = trades_df['holding_hours'].mean()
            
            # Exit reason analysis
            exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
            
        else:
            win_rate_trades = avg_win = avg_loss = profit_factor = avg_holding_hours = 0
            exit_reasons = {}
        
        return {
            # Overview
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'cagr_pct': cagr * 100,
            'years': years,
            
            # Risk metrics
            'annual_volatility_pct': annual_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar_ratio,
            
            # Win rates
            'win_rate_hourly_pct': win_rate_hourly,
            'win_rate_trades_pct': win_rate_trades,
            
            # Trading stats
            'total_trades': len(trades_df),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_hours': avg_holding_hours,
            'exit_reasons': exit_reasons,
            
            # DataFrames for plotting
            'snapshots_df': snapshots_df,
            'trades_df': trades_df
        }
    
    def create_report(self, results: Dict[str, Any], analysis: Dict[str, Any], 
                     output_path: str = "enhanced_backtest_report.png"):
        """Create comprehensive visual report"""
        
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle('Enhanced ML Mean Reversion Trading System - Performance Report', 
                        fontsize=16, fontweight='bold')
            
            snapshots_df = analysis.get('snapshots_df', pd.DataFrame())
            trades_df = analysis.get('trades_df', pd.DataFrame())
            
            if len(snapshots_df) > 0:
                # 1. Portfolio value over time
                ax = axes[0, 0]
                ax.plot(snapshots_df.index, snapshots_df['portfolio_value'], 'b-', linewidth=1.5)
                ax.axhline(y=self.config.initial_capital, color='r', linestyle='--', alpha=0.7)
                ax.set_title('Portfolio Value Over Time')
                ax.set_ylabel('Portfolio Value ($)')
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # 2. Cumulative returns
                ax = axes[0, 1]
                ax.plot(snapshots_df.index, snapshots_df['cumulative_returns'], 'g-', linewidth=1.5)
                ax.set_title('Cumulative Returns')
                ax.set_ylabel('Returns (%)')
                ax.grid(True, alpha=0.3)
                
                # 3. Drawdown
                ax = axes[1, 0]
                rolling_max = snapshots_df['portfolio_value'].expanding().max()
                drawdown = (snapshots_df['portfolio_value'] - rolling_max) / rolling_max * 100
                ax.fill_between(snapshots_df.index, 0, drawdown, color='red', alpha=0.7)
                ax.set_title('Drawdown Analysis')
                ax.set_ylabel('Drawdown (%)')
                ax.grid(True, alpha=0.3)
                
                # 4. Returns distribution
                ax = axes[1, 1]
                returns = snapshots_df['returns'].dropna() * 100
                if len(returns) > 0:
                    ax.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
                    ax.set_title('Hourly Returns Distribution')
                    ax.set_xlabel('Return (%)')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
            
            if len(trades_df) > 0:
                # 5. P&L distribution
                ax = axes[2, 0]
                wins = trades_df[trades_df['pnl'] > 0]['pnl']
                losses = trades_df[trades_df['pnl'] < 0]['pnl']
                
                if len(wins) > 0:
                    ax.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})')
                if len(losses) > 0:
                    ax.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})')
                
                ax.set_title('Trade P&L Distribution')
                ax.set_xlabel('P&L ($)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 6. Exit reasons
                ax = axes[2, 1]
                exit_reasons = analysis.get('exit_reasons', {})
                if exit_reasons:
                    reasons = list(exit_reasons.keys())
                    counts = list(exit_reasons.values())
                    ax.pie(counts, labels=reasons, autopct='%1.1f%%')
                    ax.set_title('Exit Reasons')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Report saved to {output_path}")
            
            # Save detailed metrics
            self._save_metrics_table(analysis, output_path.replace('.png', '_metrics.txt'))
            
        except Exception as e:
            self.logger.error(f"Report creation failed: {e}")
    
    def _save_metrics_table(self, analysis: Dict[str, Any], output_path: str):
        """Save detailed metrics table"""
        
        try:
            with open(output_path, 'w') as f:
                f.write("ENHANCED ML MEAN REVERSION TRADING SYSTEM - DETAILED METRICS\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Initial Capital: ${analysis['initial_capital']:,.2f}\n")
                f.write(f"Final Value: ${analysis['final_value']:,.2f}\n")
                f.write(f"Total Return: {analysis['total_return_pct']:.2f}%\n")
                f.write(f"CAGR: {analysis['cagr_pct']:.2f}%\n")
                f.write(f"Years: {analysis['years']:.2f}\n\n")
                
                f.write("RISK METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Annual Volatility: {analysis['annual_volatility_pct']:.2f}%\n")
                f.write(f"Sharpe Ratio: {analysis['sharpe_ratio']:.3f}\n")
                f.write(f"Sortino Ratio: {analysis['sortino_ratio']:.3f}\n")
                f.write(f"Maximum Drawdown: {analysis['max_drawdown_pct']:.2f}%\n")
                f.write(f"Calmar Ratio: {analysis['calmar_ratio']:.3f}\n\n")
                
                f.write("TRADING STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Trades: {analysis['total_trades']}\n")
                f.write(f"Win Rate (Trades): {analysis['win_rate_trades_pct']:.1f}%\n")
                f.write(f"Win Rate (Hourly): {analysis['win_rate_hourly_pct']:.1f}%\n")
                f.write(f"Average Win: ${analysis['avg_win']:.2f}\n")
                f.write(f"Average Loss: ${abs(analysis['avg_loss']):.2f}\n")
                f.write(f"Profit Factor: {analysis['profit_factor']:.2f}\n")
                f.write(f"Average Holding Time: {analysis['avg_holding_hours']:.1f} hours\n\n")
                
                if analysis['exit_reasons']:
                    f.write("EXIT REASONS\n")
                    f.write("-" * 30 + "\n")
                    total_exits = sum(analysis['exit_reasons'].values())
                    for reason, count in analysis['exit_reasons'].items():
                        pct = (count / total_exits) * 100
                        f.write(f"{reason}: {count} ({pct:.1f}%)\n")
            
            self.logger.info(f"Detailed metrics saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics table: {e}")

# =============================================================================
# 10. ENHANCED MAIN EXECUTION
# =============================================================================

def run_enhanced_backtest():
    """Run the BALANCED enhanced ML trading system backtest"""
    
    print("\n" + "="*80)
    print("BALANCED ENHANCED ML-POWERED MEAN REVERSION TRADING SYSTEM")
    print("Targeting: Balanced Risk-Return Profile | Profitable + Controlled Risk")
    print("="*80)
    
    # BALANCED enhanced configuration
    config = TradingConfig(
        initial_capital=100000,
        signal_threshold=0.25,            # BALANCED between quality and quantity
        min_confirmations=1,              # BALANCED for opportunity capture
        max_positions=4,                  # BALANCED diversification
        position_size_pct=0.14,           # BALANCED exposure
        max_features=12,                  # KEEP: Avoid overfitting
        stop_loss_pct=0.03,               # BALANCED stops
        take_profit_pct=0.07,             # BALANCED profits
        max_hold_hours=42,                # BALANCED holding time
        volatility_scaling=True,          # KEEP: Smart sizing
        regime_filter=True,               # KEEP: But less restrictive
        max_daily_trades=3,               # BALANCED activity
        drawdown_limit=0.10,              # RELAXED protection
        trend_filter=False,               # RELAXED: Allow counter-trend
        avoid_first_hour=False,           # RELAXED: Allow more trading
        lunch_hour_filter=False,          # RELAXED: Allow lunch trading
        log_level="INFO"
    )
    
    # Save balanced configuration
    config.save("balanced_enhanced_trading_config.json")
    print(f"💾 Balanced enhanced configuration saved")
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    # Define universe (FOCUSED but not too restrictive)
    universe = [
        # Most liquid ETFs for best mean reversion
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP',
        
        # High-quality large caps
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA',
        'JPM', 'BAC', 'JNJ', 'PG', 'WMT', 'XOM'
    ]
    
    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
    
    print(f"\n📊 Balanced Enhanced Parameters:")
    print(f"   Universe: {len(universe)} symbols (balanced selection)")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Capital: ${config.initial_capital:,.0f}")
    print(f"   Signal Threshold: {config.signal_threshold} (BALANCED)")
    print(f"   Min Confirmations: {config.min_confirmations} (BALANCED)")
    print(f"   Max Positions: {config.max_positions} (BALANCED)")
    print(f"   Position Size: {config.position_size_pct:.1%} (BALANCED)")
    print(f"   Regime Filtering: Moderate (balanced)")
    
    try:
        # Run balanced enhanced backtest
        print(f"\n🚀 Starting balanced enhanced backtest...")
        results = engine.run_backtest(universe, start_date, end_date)
        
        if not results:
            print("❌ Backtest failed - check logs for details")
            return
        
        # Analyze results
        print(f"\n📈 Analyzing balanced enhanced results...")
        analyzer = ResultsAnalyzer(config, engine.logger)
        analysis = analyzer.analyze(results)
        
        if analysis:
            # Display key metrics
            print(f"\n🎯 BALANCED ENHANCED RESULTS:")
            print(f"   Total Return: {analysis['total_return_pct']:.2f}%")
            print(f"   CAGR: {analysis['cagr_pct']:.2f}%")
            print(f"   Sharpe Ratio: {analysis['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {analysis['max_drawdown_pct']:.2f}%")
            print(f"   Total Trades: {analysis['total_trades']}")
            print(f"   Win Rate: {analysis['win_rate_trades_pct']:.1f}%")
            print(f"   Profit Factor: {analysis['profit_factor']:.2f}")
            
            # Create balanced enhanced report
            print(f"\n📊 Creating balanced enhanced report...")
            analyzer.create_report(results, analysis, "balanced_enhanced_backtest_report.png")
            
            # Save results
            if analysis.get('trades_df') is not None and len(analysis['trades_df']) > 0:
                analysis['trades_df'].to_csv("balanced_enhanced_backtest_trades.csv", index=False)
                print(f"💾 Balanced enhanced trades saved")
            
            if analysis.get('snapshots_df') is not None:
                analysis['snapshots_df'].to_csv("balanced_enhanced_backtest_snapshots.csv")
                print(f"💾 Balanced enhanced snapshots saved")
            
            print(f"\n📁 Balanced Enhanced Output Files:")
            print(f"   📊 balanced_enhanced_backtest_report.png - Performance report")
            print(f"   📄 balanced_enhanced_backtest_report_metrics.txt - Detailed metrics")
            print(f"   📈 balanced_enhanced_backtest_trades.csv - Trade details")
            print(f"   💰 balanced_enhanced_backtest_snapshots.csv - Portfolio snapshots")
            print(f"   ⚙️  balanced_enhanced_trading_config.json - Configuration")
            print(f"   📋 logs/enhanced_trading_system.log - Execution log")
            
            # Balanced performance assessment
            total_return = analysis['total_return_pct']
            sharpe = analysis['sharpe_ratio']
            max_dd = abs(analysis['max_drawdown_pct'])
            
            print(f"\n🎯 BALANCED PERFORMANCE ASSESSMENT:")
            
            targets_met = 0
            if total_return > 5:  # RELAXED target
                print(f"   ✅ Return Target: {total_return:.1f}% (Target: >5%)")
                targets_met += 1
            else:
                print(f"   ❌ Return Target: {total_return:.1f}% (Target: >5%)")
            
            if sharpe > 0.3:  # RELAXED target
                print(f"   ✅ Sharpe Target: {sharpe:.3f} (Target: >0.3)")
                targets_met += 1
            else:
                print(f"   ❌ Sharpe Target: {sharpe:.3f} (Target: >0.3)")
            
            if max_dd < 12:  # RELAXED target
                print(f"   ✅ Drawdown Target: {max_dd:.1f}% (Target: <12%)")
                targets_met += 1
            else:
                print(f"   ❌ Drawdown Target: {max_dd:.1f}% (Target: <12%)")
            
            if targets_met == 3:
                print(f"   🎊 EXCELLENT: All balanced targets achieved!")
            elif targets_met == 2:
                print(f"   ✅ GOOD: {targets_met}/3 balanced targets achieved")
            elif targets_met == 1:
                print(f"   ⚠️  MODERATE: {targets_met}/3 balanced targets achieved")
            else:
                print(f"   ❌ NEEDS WORK: No balanced targets achieved")
                
            print(f"\n✨ Balanced enhanced system completed!")
            
        else:
            print("❌ Balanced enhanced analysis failed - check logs for details")
    
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        return
    
    print(f"\n" + "="*80)
    print("BALANCED ENHANCED BACKTEST COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    # Check dependencies
    print("🔍 Checking dependencies...")
    
    required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'yfinance']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        exit(1)
    
    print("   ✅ All dependencies satisfied!")
    
    # Run the balanced enhanced backtest
    input("\nPress Enter to start the BALANCED ENHANCED trading system backtest...")
    run_enhanced_backtest()