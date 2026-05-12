import re
import joblib
import configparser
from fastapi import FastAPI
from pydantic import BaseModel

# Загружаем конфиг
config = configparser.ConfigParser()
config.read('config.ini')

# Загружаем модель и векторайзер
model = joblib.load('experiments/model.pkl')
vectorizer = joblib.load('experiments/vectorizer.pkl')

# Создаём приложение
app = FastAPI(title="Amazon Reviews Sentiment API")

# Схема входных данных
class ReviewRequest(BaseModel):
    text: str

# Функция очистки текста 
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Главный эндпоинт
@app.post('/predict')
def predict(request: ReviewRequest):
    cleaned = clean_text(request.text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return {
        'text': request.text,
        'sentiment': labels[pred]
    }

# Эндпоинт проверки что сервер живой
@app.get('/health')
def health():
    return {'status': 'ok'}