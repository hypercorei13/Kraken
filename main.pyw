import PySimpleGUI as sg
import time
import hmac
import base64
import hashlib
import urllib.parse
import requests
import json
from datetime import datetime
import threading
from typing import Dict, List
import logging
import os
from main_predict import PricePredictor

class KrakenMonitor:
    def __init__(self, query_interval: int = 60):
        """Initialize Kraken API client
        
        Args:
            query_interval: Time between price updates in seconds (default: 60)
        """
        # Set up file paths
        self.assets_dir = "assets"
        self.checked_file = os.path.join(self.assets_dir, "assets.checked")
        self.blocked_file = os.path.join(self.assets_dir, "assets.blocked")
        self.keys_file = os.path.join(self.assets_dir, "main.keys")
        
        # Create assets directory if it doesn't exist
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
        
        # Load API keys
        try:
            if os.path.exists(self.keys_file):
                with open(self.keys_file, 'r') as f:
                    keys = json.load(f)
                    self.api_key = keys.get('api_key', '')
                    self.api_secret = keys.get('api_secret', '')
            else:
                # Create default keys file
                default_keys = {
                    'api_key': 'YOUR_API_KEY_HERE',
                    'api_secret': 'YOUR_API_SECRET_HERE'
                }
                with open(self.keys_file, 'w') as f:
                    json.dump(default_keys, f, indent=4)
                self.api_key = default_keys['api_key']
                self.api_secret = default_keys['api_secret']
                self.logger.warning("Created default keys file. Please update with your API keys.")
        except Exception as e:
            self.logger.error(f"Error loading API keys: {str(e)}")
            self.api_key = ''
            self.api_secret = ''
        
        self.api_url = "https://api.kraken.com"
        self.api_version = "0"
        
        # Store query interval
        self.query_interval = query_interval
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize GUI theme and settings
        sg.theme('Black')
        self.font = ('Helvetica', 10)
        self.title_font = ('Helvetica', 12, 'bold')
        
        # Store asset balances and prices
        self.balances = {}
        self.prices = {}
        self.running = True
        
        # Price history cache (store last 5 updates for each asset)
        self.price_history = {}
        
        # Initialize assets first
        self.all_assets = {}
        self.selected_assets = set()
        self.asset_checkboxes = []  # Initialize checkbox list
        
        # Initialize blocked assets and load them
        self.blocked_assets = set()
        self.load_blocked_assets()
        
        # Now get all assets after blocked assets are loaded
        self.get_all_assets()  # Get asset info
        self.logger.info(f"Loaded {len(self.all_assets)} assets")
        
        # Finally load checked assets
        self.load_checked_assets()
        self.logger.info(f"Loaded {len(self.selected_assets)} selected assets")
        
        # Initialize price predictor AFTER other initializations
        self.predictor = PricePredictor(update_interval=query_interval)
        self.logger.info("Price predictor initialized")
    
    def _get_kraken_signature(self, urlpath: str, data: Dict) -> str:
        """Create authenticated signature for Kraken API"""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(base64.b64decode(self.api_secret),
                      message,
                      hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    def _api_request(self, endpoint: str, data: Dict = None, public: bool = True) -> Dict:
        """Make request to Kraken API"""
        if data is None:
            data = {}
        
        if not public:
            data['nonce'] = str(int(time.time() * 1000))
        
        # Remove 'public/' and 'private/' from endpoint if present
        endpoint = endpoint.replace('public/', '').replace('private/', '')
        
        # Construct correct urlpath
        if public:
            urlpath = f'/0/public/{endpoint}'
        else:
            urlpath = f'/0/private/{endpoint}'
        
        try:
            if public:
                response = requests.get(
                    f'{self.api_url}{urlpath}',
                    params=data
                )
            else:
                response = requests.post(
                    f'{self.api_url}{urlpath}',
                    data=data,
                    headers={
                        'API-Key': self.api_key,
                        'API-Sign': self._get_kraken_signature(urlpath, data)
                    }
                )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('error'):
                raise Exception(f"Kraken API error: {result['error']}")
                
            return result.get('result', {})
            
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            return {}  # Return empty dict instead of error dict
    
    def get_account_balance(self) -> Dict:
        """Get account balances"""
        try:
            result = self._api_request('Balance', public=False)  # Remove 'private/' from endpoint
            if isinstance(result, dict):
                return {k: float(v) for k, v in result.items() if float(v) > 0}
            return {}
        except Exception as e:
            self.logger.error(f"Balance error: {str(e)}")
            return {}
    
    def get_ticker(self, pairs: List[str]) -> Dict:
        """Get ticker information for pairs"""
        try:
            tradable_pairs = self._api_request('AssetPairs')
            if not isinstance(tradable_pairs, dict):
                self.logger.error("Invalid response from AssetPairs")
                return {}
            
            valid_usd_pairs = {
                pair_info['altname'] 
                for pair_name, pair_info in tradable_pairs.items() 
                if pair_info['altname'].endswith('USD') and not self.is_blocked_asset(pair_info['altname'])
            }
            self.logger.info(f"Found {len(valid_usd_pairs)} valid USD trading pairs")
            
            formatted_pairs = []
            for pair in pairs:
                if pair.endswith('USD'):
                    if self.is_blocked_asset(pair):
                        continue
                    
                    base = pair[:-3]  # Remove USD
                    # Skip special cases and invalid assets
                    if base in ['ZUSD'] or len(base) <= 1:
                        continue
                    elif base in ['XRP', 'XXRP']:
                        if 'XRPUSD' in valid_usd_pairs:
                            formatted_pairs.append('XRPUSD')
                    elif base == 'BTC':
                        formatted_pairs.append('XXBTZUSD')
                    elif f"{base}USD" in valid_usd_pairs:
                        formatted_pairs.append(f"{base}USD")
            
            # Debug logging
            self.logger.info(f"Requesting {len(formatted_pairs)} pairs")
            
            all_ticker_data = {}
            chunk_size = 100
            for i in range(0, len(formatted_pairs), chunk_size):
                chunk = formatted_pairs[i:i + chunk_size]
                result = self._api_request('Ticker', {'pair': ','.join(chunk)})
                if isinstance(result, dict) and 'error' not in result:
                    all_ticker_data.update(result)
                    
            processed_data = {}
            for pair_name, data in all_ticker_data.items():
                if pair_name == 'XXBTZUSD':
                    processed_data['BTCUSD'] = data
                else:
                    base = pair_name.replace('ZUSD', '').replace('USD', '')
                    if base.startswith('X'):
                        base = base[1:]
                    processed_data[f"{base}USD"] = data
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Ticker error: {str(e)}")
            return {}
    
    def update_data(self, window: sg.Window):
        """Update price data and GUI"""
        while self.running:
            try:
                # Get account balances
                balances = self.get_account_balance()
                if balances:
                    self.balances = balances

                # Get prices for selected assets and assets with balance
                pairs = ['BTCUSD']  # Always include Bitcoin
                pairs.extend(f"{asset}USD" for asset in self.balances.keys())
                pairs.extend(f"{asset}USD" for asset in self.selected_assets)
                pairs.append('ZUSDUSD')
                pairs = list(set(pairs))

                ticker_data = self.get_ticker(pairs)
                
                if ticker_data:
                    self.prices = {}
                    predictions = {}  # Store predictions
                    
                    for pair_name, data in ticker_data.items():
                        if pair_name == 'XRPUSD':
                            self.prices['XXRPUSD'] = float(data['c'][0])
                        else:
                            base = pair_name.replace('USD', '')
                            if base.startswith('X'):
                                base = base[1:]
                            current_price = float(data['c'][0])
                            self.prices[f"{base}USD"] = current_price
                            
                            # Get prediction for each asset
                            if base != 'ZUSD':  # Skip USD predictions
                                try:
                                    prediction = self.predictor.get_prediction(base)
                                    predictions[base] = prediction
                                    self.logger.debug(f"Prediction for {base}: {prediction}")
                                except Exception as e:
                                    self.logger.error(f"Prediction error for {base}: {str(e)}")
                                    predictions[base] = current_price
                    
                    self.prices['ZUSDUSD'] = 1.00

                    # Update GUI
                    if window:
                        window.write_event_value('-UPDATE-', {
                            'balances': self.balances,
                            'prices': self.prices,
                            'predictions': predictions,  # Add predictions to update event
                            'time': datetime.now().strftime('%H:%M:%S')
                        })
                
                time.sleep(self.query_interval)
                
            except Exception as e:
                self.logger.error(f"Update error: {str(e)}")
                time.sleep(5)
    
    def create_layout(self) -> List:
        """Create clean GUI layout with asset selection menu"""
        # Create asset selection menu with search box
        self.asset_checkboxes = []  # Store checkbox references
        
        # Sort assets and create checkboxes
        sorted_assets = sorted(self.all_assets.keys())
        self.logger.info(f"Creating checkboxes for {len(sorted_assets)} assets")
        
        for asset in sorted_assets:
            # Set default value based on selected_assets
            is_checked = asset in self.selected_assets
            checkbox = sg.Checkbox(
                asset, 
                key=f'-ASSET-{asset}-', 
                enable_events=True,
                default=is_checked,
                font=self.font
            )
            self.asset_checkboxes.append(checkbox)

        # Debug log the number of checkboxes created
        self.logger.info(f"Created {len(self.asset_checkboxes)} checkboxes")

        header = [
            [sg.Text('KRAKEN MONITOR', font=self.title_font, pad=(0,10), text_color='#00FF41'),
             sg.Push(),
             sg.Button('Exit', font=self.font, button_color=('white', 'red'), key='-EXIT-')],
            [sg.Button('Select Assets', font=self.font), 
             sg.Text('Last Update:', font=self.font), 
             sg.Text('', key='-TIME-', font=self.font, text_color='#00FF41'),
             sg.Text('BTC:', font=self.font, pad=(20,0)),
             sg.Text('', key='-BTC-PRICE-', font=self.font, text_color='white')],
            [sg.HorizontalSeparator(pad=(0,10))],
            [sg.Text('Asset', size=(8, 1), font=self.title_font, text_color='#00FF41'),
             sg.Text('Balance', size=(10, 1), font=self.title_font, text_color='#00FF41'),
             sg.Text('Price', size=(10, 1), font=self.title_font, text_color='#00FF41'),
             sg.Text('Value', size=(10, 1), font=self.title_font, text_color='#00FF41'),
             sg.Text('1h Predict', size=(12, 1), font=self.title_font, text_color='#00FF41')]
        ]

        # Pre-create asset rows with keys
        asset_rows = []
        for i in range(20):
            row = [
                sg.Text('', size=(8, 1), key=f'-ASSET{i}-', font=self.font, text_color='#00FF41', visible=False),
                sg.Text('', size=(15, 1), key=f'-BAL{i}-', font=self.font, text_color='white', visible=False),
                sg.Text('', size=(12, 1), key=f'-PRICE{i}-', font=self.font, text_color='white', visible=False),
                sg.Text('', size=(12, 1), key=f'-VALUE{i}-', font=self.font, text_color='#00FF41', visible=False),
                sg.Text('', size=(12, 1), key=f'-PREDICT{i}-', font=self.font, text_color='#FFA500', visible=False)
            ]
            asset_rows.append(row)

        # Create asset selection frame with search box and select all buttons
        asset_selection = [
            [sg.Input(size=(20,1), key='-SEARCH-', enable_events=True, font=self.font),
             sg.Text('Search', font=self.font)],
            [sg.Button('Select All', key='-SELECT-ALL-', font=self.font),
             sg.Button('Unselect All', key='-UNSELECT-ALL-', font=self.font)],
            [sg.Column(
                [[cb] for cb in self.asset_checkboxes], 
                scrollable=True, 
                vertical_scroll_only=True,
                size=(350, 200),
                key='-ASSET-LIST-'
            )]
        ]

        # Main layout
        main_layout = header + [
            [sg.Column(asset_rows, key='-ASSETS-', size=(500, 560), pad=(0,0))],
            [sg.HorizontalSeparator(pad=(0,10))],
            [sg.Text('TOTAL USD:', size=(8, 1), font=self.title_font, text_color='#00FF41'),
             sg.Text('$0.00', key='-TOTAL-', size=(15, 1), font=self.title_font, text_color='#00FF41')]
        ]

        # Final layout
        layout = [
            [sg.Column(main_layout, background_color='black', pad=(10,10))],
            [sg.Column(
                [[sg.Frame('Asset Selection', asset_selection)]], 
                background_color='black',
                pad=(10,10),
                key='-ASSET-FRAME-',
                visible=False
            )]
        ]

        return layout
    
    def update_price_history(self, asset: str, price: float):
        """Update price history for an asset"""
        if asset not in self.price_history:
            self.price_history[asset] = []
        
        history = self.price_history[asset]
        history.append(price)
        
        # Keep only last 5 prices
        if len(history) > 5:
            history.pop(0)
        
        return history
    
    def get_price_color(self, asset: str, current_price: float) -> str:
        """Determine price color based on trend"""
        history = self.price_history.get(asset, [])
        if len(history) < 2:  # Not enough data for comparison
            return 'white'
        
        avg_price = sum(history[:-1]) / len(history[:-1])  # Average of previous prices
        
        if current_price > avg_price:
            return '#00FF41'  # Green
        elif current_price < avg_price:
            return '#FF4444'  # Red
        return 'white'
    
    def get_all_assets(self) -> Dict:
        """Get information about all available assets from Kraken"""
        try:
            result = self._api_request('Assets')
            if isinstance(result, dict):  # Make sure we have a valid response
                # Filter out blocked assets and special prefixes
                self.all_assets = {
                    asset: info for asset, info in result.items()
                    if not self.is_blocked_asset(asset) and 
                    not asset.startswith('Z') and  # Filter out fiat currencies
                    not asset.startswith('X') and  # Filter out special assets
                    len(asset) > 1 and  # Filter out single character assets
                    not asset.endswith('.S')  # Filter out staked assets
                }
                
                # Remove any remaining special cases
                special_prefixes = ['XX', 'XZ', 'ZX', 'ZZ']
                self.all_assets = {
                    asset: info for asset, info in self.all_assets.items()
                    if not any(asset.startswith(prefix) for prefix in special_prefixes)
                }
                
                self.logger.info(f"Retrieved {len(self.all_assets)} non-blocked assets from Kraken")
                return self.all_assets
            self.logger.error(f"Invalid assets response: {result}")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get asset info: {str(e)}")
            return {}
    
    def load_checked_assets(self):
        """Load previously checked assets from file"""
        try:
            if os.path.exists(self.checked_file):
                with open(self.checked_file, 'r') as f:
                    self.selected_assets = set(json.load(f))
                self.logger.info(f"Loaded {len(self.selected_assets)} checked assets from file")
        except Exception as e:
            self.logger.error(f"Error loading checked assets: {str(e)}")

    def save_checked_assets(self):
        """Save currently checked assets to file"""
        try:
            with open(self.checked_file, 'w') as f:
                json.dump(list(self.selected_assets), f)
            self.logger.info(f"Saved {len(self.selected_assets)} checked assets to file")
        except Exception as e:
            self.logger.error(f"Error saving checked assets: {str(e)}")
    
    def load_blocked_assets(self):
        """Load blocked assets from file"""
        try:
            if os.path.exists(self.blocked_file):
                with open(self.blocked_file, 'r') as f:
                    self.blocked_assets = {line.strip().upper() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(self.blocked_assets)} blocked assets from file")
            else:
                # Create default blocked assets file if it doesn't exist
                default_blocked = {'GENS', 'MPL', 'ANT', 'TRIBE', 'AVT', 'PARA', 'SXP', 'WETH', 'OTP', 'PICA'}
                with open(self.blocked_file, 'w') as f:
                    f.write('\n'.join(sorted(default_blocked)))
                self.blocked_assets = default_blocked
                self.logger.info(f"Created default blocked assets file with {len(default_blocked)} assets")
        except Exception as e:
            self.logger.error(f"Error loading blocked assets: {str(e)}")
            self.blocked_assets = set()

    def is_blocked_asset(self, pair: str) -> bool:
        """Check if an asset pair contains any blocked assets"""
        # Extract base asset from pair (remove USD suffix)
        base = pair[:-3] if pair.endswith('USD') else pair
        base = base.upper()
        
        # Check if any blocked asset is contained in the base
        return any(blocked in base for blocked in self.blocked_assets)
    
    def run(self):
        """Run the monitor application"""
        # Debug log before creating window
        self.logger.info(f"Creating window with {len(self.all_assets)} assets")
        self.logger.info(f"Created {len(self.asset_checkboxes)} checkboxes")
        
        window = sg.Window(
            'Kraken Monitor',
            self.create_layout(),
            no_titlebar=True,
            grab_anywhere=True,
            alpha_channel=0.9,
            finalize=True,
            keep_on_top=True,
            background_color='black',
            margins=(0,0),
            resizable=True,  # Allow window resizing
            size=(600, 800)
        )
        
        # Store initial window size
        initial_height = window.size[1]
        
        # Store all assets and checkboxes for filtering
        all_assets = sorted(self.all_assets.keys())
        checkbox_rows = [[cb] for cb in self.asset_checkboxes]
        self.logger.info(f"Created {len(checkbox_rows)} checkbox rows")
        
        # Initialize the asset list visibility
        window['-ASSET-FRAME-'].update(visible=False)
        
        # Correct way to update Column contents
        asset_column = window['-ASSET-LIST-'].Widget
        for cb in self.asset_checkboxes:
            cb.update(visible=True)
        
        self.logger.info("Updated asset list column")
        
        update_thread = threading.Thread(target=self.update_data, args=(window,), daemon=True)
        update_thread.start()
        
        while True:
            event, values = window.read()
            
            if event in (sg.WIN_CLOSED, '-EXIT-'):
                self.running = False
                self.save_checked_assets()
                if update_thread:
                    update_thread.join(timeout=1)
                window.close()
                break
            
            elif event == '-SEARCH-':
                search_term = values['-SEARCH-'].lower()
                # Filter checkboxes based on search term
                for cb in self.asset_checkboxes:
                    cb.update(visible=search_term in cb.Text.lower())
            
            elif event == 'Select Assets':
                # Toggle asset selection frame visibility
                current_visible = window['-ASSET-FRAME-'].visible
                window['-ASSET-FRAME-'].update(visible=not current_visible)
                
                # Adjust window size based on frame visibility
                if not current_visible:  # If becoming visible
                    window.size = (600, initial_height + 300)  # Add space for asset frame
                    for cb in self.asset_checkboxes:
                        cb.update(visible=True)
                    self.logger.info("Asset list made visible")
                else:
                    window.size = (600, initial_height)  # Return to original size
            
            elif event.startswith('-ASSET-') and event != '-ASSET-FRAME-' and event != '-ASSET-LIST-':
                # Update selected assets when checkboxes are clicked
                asset = event.replace('-ASSET-', '').replace('-', '')
                if values[event]:  # If checkbox is checked
                    self.selected_assets.add(asset)
                else:
                    self.selected_assets.discard(asset)
                self.save_checked_assets()
                self.logger.info(f"Selected assets: {self.selected_assets}")
            
            elif event == '-UPDATE-':
                total_usd = 0
                # Update Bitcoin price and prediction if available
                btc_price = values['-UPDATE-']['prices'].get('BTCUSD', 0)
                btc_prediction = values['-UPDATE-']['predictions'].get('BTC', 0)
                
                if btc_price > 0:
                    # Update BTC price
                    window['-BTC-PRICE-'].update(f'${btc_price:,.2f}')
                    # Update color based on price history
                    self.update_price_history('BTC', btc_price)
                    price_color = self.get_price_color('BTC', btc_price)
                    window['-BTC-PRICE-'].update(text_color=price_color)
                    
                    # Add prediction percentage next to price
                    if btc_prediction > 0:
                        pred_change = ((btc_prediction - btc_price) / btc_price * 100)
                        pred_color = '#00FF41' if pred_change > 0 else '#FF4444'
                        window['-BTC-PRICE-'].update(
                            f'${btc_price:,.2f} ({pred_change:+.2f}%)', 
                            text_color=pred_color
                        )
                
                # Combine held assets and selected assets
                all_monitored_assets = []
                predictions = values['-UPDATE-'].get('predictions', {})
                
                # Add assets with balances
                for asset, balance in values['-UPDATE-']['balances'].items():
                    all_monitored_assets.append({
                        'asset': asset,
                        'balance': balance,
                        'has_balance': True
                    })
                
                # Add selected assets that aren't in balances
                for asset in self.selected_assets:
                    if asset not in values['-UPDATE-']['balances']:
                        all_monitored_assets.append({
                            'asset': asset,
                            'balance': 0,
                            'has_balance': False
                        })
                
                # Update each asset row
                for i in range(20):
                    if i < len(all_monitored_assets):
                        asset_info = all_monitored_assets[i]
                        asset = asset_info['asset']
                        balance = asset_info['balance']
                        price = values['-UPDATE-']['prices'].get(f"{asset}USD", 0)
                        usd_value = balance * price
                        total_usd += usd_value
                        
                        # Update price history and get color
                        self.update_price_history(asset, price)
                        price_color = self.get_price_color(asset, price)
                        
                        # Get prediction
                        prediction = predictions.get(asset, 0)
                        pred_change = ((prediction - price) / price * 100) if price > 0 else 0
                        pred_color = '#00FF41' if pred_change > 0 else '#FF4444'
                        
                        # Update row visibility and values
                        window[f'-ASSET{i}-'].update(f'{asset}:', visible=True)
                        window[f'-BAL{i}-'].update(
                            f'{balance:.8f}' if asset_info['has_balance'] else '-', 
                            visible=True
                        )
                        window[f'-PRICE{i}-'].update(f'${price:.8f}', visible=True, text_color=price_color)
                        window[f'-VALUE{i}-'].update(
                            f'${usd_value:.2f}' if asset_info['has_balance'] else '-', 
                            visible=True
                        )
                        window[f'-PREDICT{i}-'].update(
                            f'{pred_change:+.2f}%' if prediction > 0 else '-',
                            visible=True,
                            text_color=pred_color
                        )
                    else:
                        # Hide unused rows
                        window[f'-ASSET{i}-'].update(visible=False)
                        window[f'-BAL{i}-'].update(visible=False)
                        window[f'-PRICE{i}-'].update(visible=False)
                        window[f'-VALUE{i}-'].update(visible=False)
                        window[f'-PREDICT{i}-'].update(visible=False)
                
                # Update total and time
                window['-TOTAL-'].update(f'${total_usd:.2f}')
                window['-TIME-'].update(values['-UPDATE-']['time'])
            
            elif event == '-SELECT-ALL-':
                # Select all visible checkboxes
                for row in window['-ASSET-LIST-'].Rows:
                    checkbox = row[0]
                    checkbox.update(value=True)
                    asset = checkbox.Text
                    self.selected_assets.add(asset)
                self.save_checked_assets()
            
            elif event == '-UNSELECT-ALL-':
                # Unselect all visible checkboxes
                for row in window['-ASSET-LIST-'].Rows:
                    checkbox = row[0]
                    checkbox.update(value=False)
                    asset = checkbox.Text
                    self.selected_assets.discard(asset)
                self.save_checked_assets()
            
if __name__ == '__main__':
    # Configure the update interval (in seconds)
    UPDATE_INTERVAL = 10  # Change this value to adjust update frequency
    
    # Initialize and run the monitor
    monitor = KrakenMonitor(query_interval=UPDATE_INTERVAL)
    monitor.run() 