import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}

def test_predict_positive():
    response = client.post('/predict', json={'text': 'This album is absolutely amazing!'})
    assert response.status_code == 200
    assert response.json()['sentiment'] == 'positive'

def test_predict_negative():
    response = client.post('/predict', json={'text': 'Terrible quality, waste of money.'})
    assert response.status_code == 200
    assert response.json()['sentiment'] == 'negative'

def test_predict_empty():
    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 200
    assert 'sentiment' in response.json()

def test_predict_returns_text():
    text = 'Great music!'
    response = client.post('/predict', json={'text': text})
    assert response.json()['text'] == text