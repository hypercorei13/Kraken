import pytest
import os
import json
from unittest.mock import Mock, patch
from main import KrakenMonitor

@pytest.fixture
def kraken_monitor():
    """Create a KrakenMonitor instance for testing"""
    with patch('main.PricePredictor'):  # Mock the predictor
        monitor = KrakenMonitor(query_interval=10)
        yield monitor

@pytest.fixture
def sample_api_response():
    """Sample API response data"""
    return {
        'result': {
            'XXBTZUSD': {
                'c': ['50000.0', '1.0']
            },
            'XETHZUSD': {
                'c': ['3000.0', '1.0']
            }
        }
    }

def test_init(kraken_monitor):
    """Test initialization of KrakenMonitor"""
    assert kraken_monitor.query_interval == 10
    assert isinstance(kraken_monitor.balances, dict)
    assert isinstance(kraken_monitor.prices, dict)
    assert isinstance(kraken_monitor.all_assets, dict)
    assert isinstance(kraken_monitor.selected_assets, set)

def test_get_kraken_signature(kraken_monitor):
    """Test signature generation"""
    urlpath = '/0/private/Balance'
    data = {'nonce': '1234567890'}
    signature = kraken_monitor._get_kraken_signature(urlpath, data)
    assert isinstance(signature, str)
    assert len(signature) > 0

@patch('requests.get')
def test_api_request_public(mock_get, kraken_monitor, sample_api_response):
    """Test public API request"""
    mock_get.return_value.json.return_value = sample_api_response
    mock_get.return_value.raise_for_status = Mock()
    
    result = kraken_monitor._api_request('Ticker', {'pair': 'XXBTZUSD'})
    assert isinstance(result, dict)
    assert 'XXBTZUSD' in sample_api_response['result']

@patch('requests.post')
def test_api_request_private(mock_post, kraken_monitor):
    """Test private API request"""
    mock_post.return_value.json.return_value = {'result': {'balance': '1.0'}}
    mock_post.return_value.raise_for_status = Mock()
    
    result = kraken_monitor._api_request('Balance', public=False)
    assert isinstance(result, dict)

def test_update_price_history(kraken_monitor):
    """Test price history updates"""
    asset = 'BTC'
    price = 50000.0
    history = kraken_monitor.update_price_history(asset, price)
    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0] == price

def test_get_price_color(kraken_monitor):
    """Test price color determination"""
    asset = 'BTC'
    # Add some history
    kraken_monitor.update_price_history(asset, 50000.0)
    kraken_monitor.update_price_history(asset, 51000.0)
    
    color = kraken_monitor.get_price_color(asset, 52000.0)
    assert isinstance(color, str)
    assert color in ['#00FF41', '#FF4444', 'white']

def test_is_blocked_asset(kraken_monitor):
    """Test blocked asset checking"""
    # Add a blocked asset
    kraken_monitor.blocked_assets.add('TEST')
    assert kraken_monitor.is_blocked_asset('TESTUSD')
    assert not kraken_monitor.is_blocked_asset('BTCUSD')

@patch('builtins.open', create=True)
def test_load_checked_assets(mock_open, kraken_monitor):
    """Test loading checked assets"""
    mock_open.return_value.__enter__.return_value.read.return_value = '["BTC", "ETH"]'
    kraken_monitor.load_checked_assets()
    assert 'BTC' in kraken_monitor.selected_assets
    assert 'ETH' in kraken_monitor.selected_assets

@patch('builtins.open', create=True)
def test_save_checked_assets(mock_open, kraken_monitor):
    """Test saving checked assets"""
    kraken_monitor.selected_assets = {'BTC', 'ETH'}
    kraken_monitor.save_checked_assets()
    mock_open.assert_called_once()

@patch('requests.get')
def test_get_ticker(mock_get, kraken_monitor, sample_api_response):
    """Test ticker data retrieval"""
    mock_get.return_value.json.return_value = sample_api_response
    mock_get.return_value.raise_for_status = Mock()
    
    result = kraken_monitor.get_ticker(['BTCUSD'])
    assert isinstance(result, dict)
    assert 'BTCUSD' in result or 'XXBTZUSD' in result

def test_create_layout(kraken_monitor):
    """Test GUI layout creation"""
    layout = kraken_monitor.create_layout()
    assert isinstance(layout, list)
    assert len(layout) > 0 