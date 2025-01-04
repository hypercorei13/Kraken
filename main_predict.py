import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import requests
import time
import logging
from typing import Dict, List, Tuple
import threading
import queue
import os
import json
from datetime import datetime, timedelta
import pickle

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_layers=3):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,  # Number of features (14 indicators + price)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Decode the hidden state
        out = self.fc(context)
        return out

class PricePredictor:
    def __init__(self, update_interval: int = 60):
        """Initialize price predictor"""
        self.update_interval = update_interval
        self.models = {}
        self.scalers = {}
        self.price_history = {}
        self.predictions = {}
        self.running = True
        
        # Configure CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configure detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('price_predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"PricePredictor initialized with device: {self.device}")
        
        # Set up data directory
        self.data_dir = "price_data"
        self.ensure_directories()
        
        # Archive old data before starting
        self.archive_old_data()
        
        # Queue for receiving new price data
        self.price_queue = queue.Queue()
        
        # Load existing data
        self.load_saved_data()
        
        # Start prediction thread
        self.predict_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.predict_thread.start()

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "models"),
            os.path.join(self.data_dir, "scalers"),
            os.path.join(self.data_dir, "history")
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")

    def get_asset_files(self, asset: str) -> Tuple[str, str, str]:
        """Get file paths for asset data"""
        model_file = os.path.join(self.data_dir, "models", f"{asset}_model.pkl")
        scaler_file = os.path.join(self.data_dir, "scalers", f"{asset}_scaler.pkl")
        history_file = os.path.join(self.data_dir, "history", f"{asset}_history.json")
        return model_file, scaler_file, history_file

    def save_asset_data(self, asset: str):
        """Save asset data to files"""
        model_file, scaler_file, history_file = self.get_asset_files(asset)
        
        try:
            # Save model if exists
            if asset in self.models:
                # Save only the state dict
                torch.save(self.models[asset].state_dict(), model_file)
                self.logger.debug(f"Saved model for {asset}")
            
            # Save scaler if exists
            if asset in self.scalers:
                # Save only the scale and min/max values
                scaler_data = {
                    'scale_': self.scalers[asset].scale_,
                    'min_': self.scalers[asset].min_,
                    'data_min_': self.scalers[asset].data_min_,
                    'data_max_': self.scalers[asset].data_max_,
                    'data_range_': self.scalers[asset].data_range_
                }
                with open(scaler_file, 'wb') as f:
                    pickle.dump(scaler_data, f)
                self.logger.debug(f"Saved scaler for {asset}")
            
            # Save price history if exists
            if asset in self.price_history:
                history_data = {
                    'timestamp': datetime.now().isoformat(),
                    'prices': self.price_history[asset][-120:]
                }
                with open(history_file, 'w') as f:
                    json.dump(history_data, f)
                self.logger.debug(f"Saved price history for {asset}")
                
        except Exception as e:
            self.logger.error(f"Error saving data for {asset}: {str(e)}")

    def load_saved_data(self):
        """Load saved data for all assets"""
        try:
            model_files = os.listdir(os.path.join(self.data_dir, "models"))
            assets = set(f.split('_')[0] for f in model_files if f.endswith('_model.pkl'))
            
            for asset in assets:
                model_file, scaler_file, history_file = self.get_asset_files(asset)
                
                try:
                    # Load model
                    if os.path.exists(model_file):
                        model = LSTMPredictor().to(self.device)
                        model.load_state_dict(torch.load(model_file))
                        model.eval()
                        self.models[asset] = model
                    
                    # Load scaler
                    if os.path.exists(scaler_file):
                        with open(scaler_file, 'rb') as f:
                            scaler_data = pickle.load(f)
                            scaler = MinMaxScaler()
                            # Manually restore scaler state
                            scaler.scale_ = scaler_data['scale_']
                            scaler.min_ = scaler_data['min_']
                            scaler.data_min_ = scaler_data['data_min_']
                            scaler.data_max_ = scaler_data['data_max_']
                            scaler.data_range_ = scaler_data['data_range_']
                            self.scalers[asset] = scaler
                    
                    # Load price history
                    if os.path.exists(history_file):
                        with open(history_file, 'r') as f:
                            data = json.load(f)
                            saved_time = datetime.fromisoformat(data['timestamp'])
                            if datetime.now() - saved_time < timedelta(hours=2):
                                self.price_history[asset] = data['prices']
                                
                except Exception as e:
                    self.logger.error(f"Error loading data for {asset}: {str(e)}")
                    # Delete corrupted files
                    for file in [model_file, scaler_file, history_file]:
                        if os.path.exists(file):
                            try:
                                os.remove(file)
                                self.logger.info(f"Deleted corrupted file: {file}")
                            except Exception as e:
                                self.logger.error(f"Could not delete file {file}: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error in load_saved_data: {str(e)}")

    def _get_historical_data(self, asset: str) -> List[float]:
        """Get historical minute data for asset from Kraken"""
        self.logger.info(f"Fetching historical data for {asset}")
        try:
            # Try different pair formats
            pair_formats = [
                f"{asset}USD",      # Standard format
                f"X{asset}ZUSD",    # Special format for some major pairs
                f"XX{asset}ZUSD",   # Format for BTC
                asset + "USD",      # Uppercase format
                f"{asset}ZUSD"      # Alternative format
            ]
            
            # Special case for BTC
            if asset == "BTC":
                pair_formats.insert(0, "XXBTZUSD")  # Add BTC's special format first
            
            for pair in pair_formats:
                url = f"https://api.kraken.com/0/public/OHLC"
                params = {
                    "pair": pair,
                    "interval": 1,  # 1 minute intervals
                    "since": int(time.time() - 86400)  # Last 24 hours
                }
                
                self.logger.debug(f"Trying pair format: {pair}")
                response = requests.get(url, params=params)
                data = response.json()
                
                if "error" not in data or not data["error"]:
                    if "result" in data and data["result"]:
                        # Find the correct key in results
                        result_key = next((k for k in data["result"].keys() if k != "last"), None)
                        if result_key and data["result"][result_key]:
                            prices = [float(candle[4]) for candle in data["result"][result_key]]
                            self.logger.info(f"Retrieved {len(prices)} historical prices for {asset}")
                            return prices
            
            self.logger.error(f"No data found for {asset} after trying multiple formats")
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {asset}: {str(e)}")
            return []

    def _calculate_technical_indicators(self, prices: np.array) -> np.array:
        """Calculate technical indicators for price data"""
        try:
            df = pd.DataFrame(prices, columns=['close'])
            
            # MACD (Moving Average Convergence Divergence)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = macd - signal
            
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma + (std * 2)
            df['bb_lower'] = sma - (std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
            
            # Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Price Rate of Change
            df['roc'] = df['close'].pct_change(periods=12) * 100
            
            # Average True Range (ATR)
            high = df['close'].rolling(2).max()
            low = df['close'].rolling(2).min()
            tr = high - low
            df['atr'] = tr.rolling(window=14).mean()
            
            # Momentum
            df['momentum'] = df['close'].diff(14)
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Return numpy array of features
            return df.values
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return np.zeros((len(prices), 14))  # Return correct number of features

    def _create_model(self, asset: str) -> Tuple[LSTMPredictor, MinMaxScaler]:
        """Create and train a new model for an asset"""
        try:
            self.logger.info(f"Creating new model for {asset}")
            
            # Get historical data
            prices = self._get_historical_data(asset)
            if not prices:
                self.logger.warning(f"No historical data available for {asset}")
                return None, None
            
            # Create and fit scaler first
            scaler = MinMaxScaler()
            prices_2d = np.array(prices).reshape(-1, 1)
            scaler.fit(prices_2d)
            
            # Calculate technical indicators and scale them
            features = self._calculate_technical_indicators(prices_2d)
            
            # Prepare training data
            X, y = [], []
            for i in range(60, len(features)):
                X.append(features[i-60:i])
                y.append(prices[i])
            
            if not X:
                self.logger.warning(f"Insufficient data to train model for {asset}")
                return None, None
            
            # Convert to PyTorch tensors
            X = torch.FloatTensor(np.array(X)).to(self.device)
            y = torch.FloatTensor(np.array(y)).to(self.device)
            
            # Create model
            model = LSTMPredictor(input_size=features.shape[1]).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Train model
            model.train()
            batch_size = 32
            epochs = 100
            
            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_y = y[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    self.logger.debug(f"{asset} - Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X):.4f}")
            
            self.logger.info(f"Successfully trained model for {asset}")
            return model, scaler
            
        except Exception as e:
            self.logger.error(f"Error creating model for {asset}: {str(e)}")
            return None, None

    def _predict_price(self, asset: str, current_price: float) -> float:
        """Predict price for asset"""
        try:
            # Create model if it doesn't exist
            if asset not in self.models:
                self.logger.info(f"No existing model found for {asset}, creating new one...")
                model, scaler = self._create_model(asset)
                if model is None:
                    self.logger.warning(f"Failed to create model for {asset}")
                    return current_price
                self.models[asset] = model
                self.scalers[asset] = scaler
            
            # Update price history
            if asset not in self.price_history:
                self.price_history[asset] = []
            self.price_history[asset].append(current_price)
            if len(self.price_history[asset]) > 120:
                self.price_history[asset].pop(0)
            
            # Not enough history for prediction
            if len(self.price_history[asset]) < 60:
                self.logger.debug(f"{asset}: Insufficient history ({len(self.price_history[asset])}/60 points)")
                return current_price
            
            # Prepare data for prediction
            prices = np.array(self.price_history[asset][-60:]).reshape(-1, 1)
            features = self._calculate_technical_indicators(prices)
            
            # Debug log the shapes
            self.logger.debug(f"Features shape before reshape: {features.shape}")
            
            # Ensure we have the correct number of features and timepoints
            features = features[-60:]  # Take last 60 timepoints
            
            # Reshape to (batch_size, sequence_length, features)
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            self.logger.debug(f"Input tensor shape: {X.shape}")
            
            # Make prediction
            self.models[asset].eval()
            with torch.no_grad():
                prediction = self.models[asset](X)
                prediction = prediction.cpu().numpy()[0][0]
                
            # Scale prediction back to price
            if prediction <= 0:
                prediction = current_price
            
            self.logger.info(f"{asset}: Current=${current_price:.2f}, Predicted=${prediction:.2f}, "
                           f"Change={((prediction-current_price)/current_price)*100:.2f}%")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting price for {asset}: {str(e)}")
            return current_price

    def _prediction_loop(self):
        """Main prediction loop"""
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes
        
        while self.running:
            try:
                # Process queue
                predictions_made = 0
                while not self.price_queue.empty():
                    asset, price = self.price_queue.get_nowait()
                    self.logger.debug(f"Processing {asset} from queue, price=${price:.4f}")
                    prediction = self._predict_price(asset, price)
                    self.predictions[asset] = prediction
                    predictions_made += 1
                
                if predictions_made > 0:
                    self.logger.info(f"Made {predictions_made} predictions this cycle")
                else:
                    self.logger.debug("No predictions made this cycle")
                
                # Save data periodically
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    self.logger.info("Performing periodic data save")
                    for asset in self.models.keys():
                        self.save_asset_data(asset)
                    last_save_time = current_time
                    self.logger.info("Periodic save completed")
                    
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {str(e)}")
                time.sleep(5)

    def add_price(self, asset: str, price: float):
        """Add new price data for prediction"""
        self.logger.debug(f"Adding price for {asset}: ${price:.4f}")
        self.price_queue.put((asset, price))

    def get_prediction(self, asset: str) -> float:
        """Get price prediction for an asset"""
        try:
            # Check if we need historical data
            if asset not in self.price_history or len(self.price_history[asset]) < 60:
                prices = self._get_historical_data(asset)
                if not prices:
                    return 0.0
                self.price_history[asset] = prices
            
            # Create or get model
            if asset not in self.models or asset not in self.scalers:
                self.logger.debug(f"Creating new model for {asset}")
                model, scaler = self._create_model(asset)
                if model is None or scaler is None:
                    return 0.0
                self.models[asset] = model
                self.scalers[asset] = scaler
            
            # Prepare input data
            prices = np.array(self.price_history[asset][-60:]).reshape(-1, 1)
            if len(prices) < 60:
                self.logger.warning(f"Insufficient price history for {asset}: {len(prices)}/60")
                return 0.0
            
            # Calculate features
            features = self._calculate_technical_indicators(prices)
            if features.shape[0] < 60:
                self.logger.warning(f"Insufficient feature data for {asset}")
                return 0.0
            
            # Scale and reshape features
            X = torch.FloatTensor(features[-60:].reshape(1, 60, -1)).to(self.device)
            
            # Make prediction
            self.models[asset].eval()
            with torch.no_grad():
                prediction = self.models[asset](X)
                prediction = prediction.cpu().numpy()[0][0]
                
            # Scale prediction back to price
            current_price = self.price_history[asset][-1]
            if prediction <= 0:
                prediction = current_price
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting price for {asset}: {str(e)}")
            return 0.0

    def stop(self):
        """Stop the predictor and save data"""
        self.logger.info("Stopping price predictor")
        self.running = False
        for asset in self.models.keys():
            self.save_asset_data(asset)
        if self.predict_thread.is_alive():
            self.predict_thread.join(timeout=1)
        self.logger.info("Price predictor stopped") 

    def archive_old_data(self):
        """Archive old model files before deleting them"""
        try:
            # Create archive directory with timestamp
            archive_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_dir = os.path.join(self.data_dir, f"archive_{archive_time}")
            os.makedirs(archive_dir, exist_ok=True)
            
            # Create subdirectories
            for subdir in ["models", "scalers", "history"]:
                os.makedirs(os.path.join(archive_dir, subdir), exist_ok=True)
            
            # Move files to archive
            files_moved = 0
            for subdir in ["models", "scalers", "history"]:
                src_dir = os.path.join(self.data_dir, subdir)
                dst_dir = os.path.join(archive_dir, subdir)
                
                if os.path.exists(src_dir):
                    for file in os.listdir(src_dir):
                        src_file = os.path.join(src_dir, file)
                        dst_file = os.path.join(dst_dir, file)
                        try:
                            if os.path.exists(dst_file):
                                os.remove(dst_file)  # Remove existing file in archive if it exists
                            os.rename(src_file, dst_file)
                            files_moved += 1
                        except Exception as e:
                            self.logger.error(f"Error moving file {file}: {str(e)}")
            
            self.logger.info(f"Archived {files_moved} files to {archive_dir}")
            
        except Exception as e:
            self.logger.error(f"Error archiving old data: {str(e)}") 