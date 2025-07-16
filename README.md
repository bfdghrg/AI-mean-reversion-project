Features
Targets Sharpe Ratio > 0.5, CAGR > 10%, Max Drawdown < 10%

Machine Learning model (Random Forest) combined with robust rule-based signals

Flexible risk management: stop-loss, take-profit, trailing stops

Market regime and volatility filters

Sector exposure controls & position sizing with volatility scaling

Backtesting engine with logging, analytics, and visual reports

Modular design with components for data, features, models, portfolio, and results

🛠 How it Works
1️⃣ Data Collection

Uses Yahoo Finance hourly data (yfinance) with caching for performance.

2️⃣ Feature Engineering

Computes technical indicators, z-scores, RSI, Bollinger Bands, volatility regimes, and more.

3️⃣ Signal Generation

Combines ML predictions and rule-based signals, filtered by market regime and confirmations.

4️⃣ Portfolio Management

Manages positions, risk controls, sector limits, drawdown constraints, and trade logging.

5️⃣ Backtesting

Runs simulations over a specified universe & period, generates trades, and logs metrics.

6️⃣ Results Analysis

Outputs key metrics (CAGR, Sharpe, Drawdown, Win Rate), and creates plots & reports.

⚙️ Adjusting Settings
All configuration is centralized in the TradingConfig dataclass in advancedmeanreversion.py.

You can adjust parameters by editing the TradingConfig or passing your own when initializing.
Below are some key parameters you can tune:

Category	Parameter	Description	Default
📅 Data	lookback_days	How many days of historical data to fetch	500
📈 Signals	signal_threshold	Minimum signal strength to trade	0.25
📈 Signals	min_confirmations	# of independent confirmations required	1
🧪 ML Model	max_features	Max # of features to use	12
💰 Risk Management	initial_capital	Starting capital	100,000
💰 Risk Management	stop_loss_pct	Stop-loss as % of entry price	0.03
💰 Risk Management	take_profit_pct	Take-profit as % of entry price	0.07
📊 Portfolio	max_positions	Max open positions at once	4
📊 Portfolio	position_size_pct	% of capital to allocate per position	0.14
🧹 Filters	volatility_filter	Enable volatility regime filter	True
🧹 Filters	trend_filter	Enable trend filter	False
📄 Logging	log_level	Logging level (INFO, DEBUG, etc.)	INFO

Example: Changing Config
python
Copy
Edit
from advancedmeanreversion import TradingConfig

my_config = TradingConfig(
    initial_capital=50000,
    signal_threshold=0.3,
    max_positions=3
)
Or save your custom config to JSON:

python
Copy
Edit
my_config.save("my_config.json")
And load it later:

python
Copy
Edit
my_config = TradingConfig.load("my_config.json")

📋 How to Run
bash
Copy
Edit
python advancedmeanreversion.py
By default it runs run_enhanced_backtest() with the balanced configuration.
To use your own config or modify the symbols and date range, edit the run_enhanced_backtest() function at the bottom of the file.

📝 Output
Logs are written to logs/enhanced_trading_system.log

Backtest results and plots are saved in the working directory

Metrics report saved as .txt alongside .png performance chart

🔗 Dependencies
Python ≥3.7

pandas, numpy, scikit-learn, matplotlib, seaborn, yfinance

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
