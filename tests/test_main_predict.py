import pytest
import numpy as np
from unittest.mock import Mock, patch
from main_predict import PricePredictor

@pytest.fixture
def price_predictor():
    """Create a PricePredictor instance for testing"""
    return PricePredictor(update_interval=10)

@pytest.fixture
def sample_historical_data():
    """Sample historical price data"""
    return [100.0 + i for i in range(120)]  # 120 increasing prices

def test_init(price_predictor):
    """Test initialization of PricePredictor"""
    assert price_predictor.update_interval == 10
    assert isinstance(price_predictor.models, dict)
    assert isinstance(price_predictor.scalers, dict)

@patch('requests.get')
def test_get_historical_data(mock_get, price_predictor):
    """Test historical data retrieval"""
    mock_response = {
        'result': {
            'XXBTZUSD': [[0, "50000.0", "50100.0", "49900.0", "50000.0", "100.0", "5000000.0", 100]]
        }
    }
    mock_get.return_value.json.return_value = mock_response
    
    data = price_predictor._get_historical_data('BTC')
    assert isinstance(data, list)
    assert len(data) > 0

def test_calculate_technical_indicators(price_predictor, sample_historical_data):
    """Test technical indicator calculation"""
    prices = np.array(sample_historical_data)
    features = price_predictor._calculate_technical_indicators(prices)
    assert isinstance(features, np.ndarray)
    assert len(features) > 0

def test_prepare_data(price_predictor, sample_historical_data):
    """Test data preparation"""
    X, y = price_predictor._prepare_data(sample_historical_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)

def test_create_model(price_predictor):
    """Test model creation"""
    model, scaler = price_predictor._create_model('BTC')
    assert model is not None
    assert scaler is not None

@patch('main_predict.PricePredictor._get_historical_data')
def test_get_prediction(mock_get_historical, price_predictor, sample_historical_data):
    """Test price prediction"""
    mock_get_historical.return_value = sample_historical_data
    
    prediction = price_predictor.get_prediction('BTC')
    assert isinstance(prediction, float)
    assert prediction > 0

def test_normalize_data(price_predictor):
    """Test data normalization"""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = price_predictor._create_scaler(data)
    normalized = price_predictor._normalize_data(data, scaler)
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == data.shape

def test_create_scaler(price_predictor):
    """Test scaler creation"""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = price_predictor._create_scaler(data)
    assert scaler is not None 