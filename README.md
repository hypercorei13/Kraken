# Kraken Trading Assistant
![Screenshot 2025-01-04 172305](https://github.com/user-attachments/assets/a8490603-30bd-47ab-95bd-c6d451a064e6)

NOTE: To use this software you will need your own Kraken API Keys installed in the file assets/main.keys.
api_key = "abcdefghijklmnopqrstuvwxyz0123456789"
api_secret = "abcdefghijklmnopqrstuvwxyz0123456789"

You will also need a FREE subsccription to PySimpleGUI. No files are installed in your machine. This is the safest Python GUI method that I am aware of: https://www.pysimplegui.com/

This README provides a comprehensive overview of the project, its implementation, and technical details. 
You may want to add: Versatility for connecting to other exchanges and/or data sources. 

These tools are designed to assist TOKEN Traders who have accounts on Kraken Crypto Exchange make trading decisions and monitor account balances. DYOR
A real-time cryptocurrency trading assistant that monitors Kraken exchange prices and provides AI-powered price predictions using LSTM neural networks.

## Overview

This project provides a GUI-based trading assistant that:
- Monitors real-time cryptocurrency prices from Kraken
- Displays account balances and portfolio value
- Predicts future price movements using deep learning
- Supports GPU acceleration for neural network computations
- Automatically saves and loads trained models

## Technology Stack

- **Python 3.10 - Core programming language
- **PyTorch** - Deep learning framework for LSTM implementation
- **PySimpleGUI** - GUI framework
- **Kraken API** - Real-time market data and trading
- **NumPy/Pandas** - Data processing
- **scikit-learn** - Data preprocessing
- **numpy==1.26.4** - Data preprocessing
## Project Structure

├── main.pyw # Main GUI application
├── main_predict.py # Price prediction engine
├── price_predictor.log # Application logs
└── price_data/ # Model and data storage
├── models/ # Saved LSTM models
├── scalers/ # Data scalers
├── history/ # Price history
└── archive_/ # Archived models


## Key Components

### GUI Application (main.pyw)
- Real-time price updates
- Account balance display
- Asset selection menu
- Price prediction display
- Portfolio value tracking

### Price Predictor (main_predict.py)
- LSTM neural network implementation
- Historical data processing
- Real-time price predictions
- Model persistence
- GPU acceleration support

## AI Implementation

The price prediction engine uses:
- Long Short-Term Memory (LSTM) neural networks
- 60-minute input sequences
- 1-hour price predictions
- MinMax scaling for data normalization
- Dropout layers for regularization
- Adam optimizer with MSE loss

## Development Platform

This project was developed using:
- [Cursor IDE](https://cursor.sh/) - AI-powered code editor
- Claude 3.5 Sonnet - AI development assistant
- Windows 11 development environment

## Setup

1. Install required packages:

pip install torch numpy==1.26.4 pandas scikit-learn PySimpleGUI requests


pythonw main.pyw


## Features

- Real-time price monitoring
- AI-powered price predictions
- Portfolio tracking
- Asset selection
- GPU acceleration
- Automatic model training
- Historical data analysis
- Model persistence
- Detailed logging

## Technical Details

### LSTM Architecture
- Input size: 60 minutes
- Hidden layers: 64 units
- Number of layers: 2
- Dropout: 0.2
- Training epochs: 50
- Batch size: 32

Price Range-Specific Models
The system uses different LSTM model configurations based on asset price ranges:
-Micro Range (< $0.01)
  Smaller network (64 hidden units, 2 layers)
  Higher learning rate (0.001)
  Less dropout (0.2)
-Low Range ($0.01 - $1.00)
  Medium-small network (96 hidden units, 2 layers)
  Standard learning rate (0.001)
  Moderate dropout (0.3)
-Medium Range ($1.00 - $100)
  Medium network (128 hidden units, 3 layers)
  Standard learning rate (0.001)
-Moderate dropout (0.3)
  High Range ($100 - $1000)
  Large network (160 hidden units, 3 layers)
  Lower learning rate (0.0005)
  Higher dropout (0.4)
-Mega Range (> $1000)
  Largest network (192 hidden units, 4 layers)
  Lower learning rate (0.0005)
  Higher dropout (0.4)
Each model is optimized for its price range's characteristics, with larger and more complex models for higher-priced assets where precision is more critical.

### Data Processing
- 1-minute candle data
- 24-hour historical data
- MinMax scaling
- Real-time updates
- Automatic retraining

- Technical Indicators
  The system calculates several technical indicators for each asset:
  MACD (Moving Average Convergence Divergence)
  Uses 12/26 day EMAs and 9-day signal line
  Includes MACD histogram for momentum analysis
  RSI (Relative Strength Index)
  14-period RSI for overbought/oversold conditions
  Helps identify potential trend reversals
  Bollinger Bands
  20-period moving average with 2 standard deviation bands
  Includes BB width for volatility measurement
  Moving Averages
  SMA-5 (short term trends)
  SMA-20 (medium term trends)
  SMA-50 (long term trends)
  Price Action Indicators
  Rate of Change (ROC) - 12-period momentum
  Average True Range (ATR) - 14-period volatility
  Momentum - 14-period price change

### Model Management
- Automatic saving/loading
- Model archiving
- Corruption detection
- Version control

## Logging

Detailed logging is available in `price_predictor.log`, including:
- Price updates
- Prediction results
- Model training
- Error handling
- API interactions

## Performance

The application is optimized for:
- Real-time processing
- GPU acceleration
- Memory efficiency
- Error resilience
- Data persistence

## Future Improvements

Potential areas for enhancements
- Trading automation
- Enhanced visualization
- Portfolio optimization
- Risk management

## Credits

Developed by Claude 3.5 Sonnet (Anthropic) using the Cursor IDE platform.

## License

MIT License
![Screenshot 2025-01-04 172245](https://github.com/user-attachments/assets/06e0d0ef-0189-4ed3-a0d3-3b7e94775cf3)

