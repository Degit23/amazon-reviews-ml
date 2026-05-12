import pandas as pd
import json
import re
import joblib
import os
import configparser
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

config = configparser.ConfigParser()
config.read('config.ini')

MAX_ITER = int(config['model']['max_iter'])
C = float(config['model']['C'])
MAX_FEATURES = int(config['model']['max_features'])
TEST_SIZE = float(config['model']['test_size'])
RANDOM_STATE = int(config['model']['random_state'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

def train():
    records = []
    with open('data/reviews_Digital_Music_5.json', 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    df = df.dropna(subset=['reviewText', 'overall'])
    df['label'] = df['overall'].apply(get_sentiment)
    df['clean_text'] = df['reviewText'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'],
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(C=C, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

    os.makedirs('experiments', exist_ok=True)
    joblib.dump(model, 'experiments/model.pkl')
    joblib.dump(vectorizer, 'experiments/vectorizer.pkl')
    print("Model saved!")

if __name__ == '__main__':
    train()