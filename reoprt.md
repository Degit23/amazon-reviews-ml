# Лабораторная работа №1
## Классический жизненный цикл разработки моделей машинного обучения

**Дисциплина:** Инфраструктура больших данных  
**Университет:** Национальный исследовательский университет ИТМО  
**Семестр:** Весна 2026  
**Вариант:** 14 — Amazon Reviews (https://jmcauley.ucsd.edu/data/amazon/)

---

## Цель работы

Получить навыки разработки CI/CD pipeline для ML моделей с достижением метрик моделей и качества.

---

## Ссылки

| Ресурс | Ссылка |
|--------|--------|
| GitHub репозиторий | https://github.com/Degit23/amazon-reviews-ml |
| Docker образ на DockerHub | https://hub.docker.com/r/deg1t23/amazon-reviews-ml |

---

## Ход работы

### 1. Репозиторий GitHub

Создан публичный репозиторий `amazon-reviews-ml` на GitHub. Разработка велась в ветке `dev`. Регулярно проводились коммиты с осмысленными сообщениями.

Структура репозитория:
```
amazon-reviews-ml/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── notebooks/
│   └── model.ipynb
├── src/
│   ├── app.py
│   └── train.py
├── tests/
│   └── test_app.py
├── data/
├── experiments/
├── config.ini
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── dev_sec_ops.yml
```

---

### 2. Подготовка данных

**Датасет:** Amazon Digital Music Reviews (~64 706 отзывов)  
**Источник:** https://jmcauley.ucsd.edu/data/amazon/

Использованные поля:
- `reviewText` — текст отзыва (входные данные)
- `overall` — рейтинг 1-5 (целевая переменная)

**Предобработка:**
- Удаление пустых строк (`dropna`)
- Преобразование рейтинга в 3 класса: negative (1-2), neutral (3), positive (4-5)
- Очистка текста: приведение к нижнему регистру, удаление знаков препинания и цифр

**Распределение классов:**
```
negative:  5 801  (9%)
neutral:   6 789  (10%)
positive: 52 116  (81%)
```

---

### 3. ML модель

**Алгоритм:** Логистическая регрессия (LogisticRegression)  
**Векторизация текста:** TF-IDF (TfidfVectorizer)

**Гиперпараметры (из config.ini):**
```ini
[model]
max_iter = 1000
C = 1.0
max_features = 10000
test_size = 0.2
random_state = 42
```

**Результаты обучения:**
```
Accuracy: 0.8524

              precision    recall  f1-score   support
    negative       0.76      0.47      0.58      1200
     neutral       0.53      0.21      0.30      1350
    positive       0.87      0.98      0.92     10392
    accuracy                           0.85     12942
```

---

### 4. Конвертация в .py скрипты и API сервис

Модель конвертирована из Jupyter ноутбука в два Python скрипта:

**`src/train.py`** — скрипт обучения модели:
- Загружает данные из JSON файла
- Предобрабатывает текст
- Обучает модель
- Сохраняет `model.pkl` и `vectorizer.pkl` в папку `experiments/`

**`src/app.py`** — FastAPI сервис с эндпоинтами:
- `POST /predict` — принимает текст отзыва, возвращает sentiment
- `GET /health` — проверка работоспособности сервиса

Пример запроса:
```json
POST /predict
{
  "text": "This album is absolutely amazing!"
}
```

Пример ответа:
```json
{
  "text": "This album is absolutely amazing!",
  "sentiment": "positive"
}
```

Запуск сервиса:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

### 5. Тестирование

Использован фреймворк **pytest**. Тесты расположены в `tests/test_app.py`.

**Написано 5 тестов:**

| Тест | Описание | Результат |
|------|----------|-----------|
| `test_health` | Проверка эндпоинта /health | PASSED |
| `test_predict_positive` | Позитивный отзыв → positive | PASSED |
| `test_predict_negative` | Негативный отзыв → negative | PASSED |
| `test_predict_empty` | Пустой текст не вызывает ошибку | PASSED |
| `test_predict_returns_text` | API возвращает исходный текст | PASSED |

**Результат запуска:**
```
5 passed in 2.14s
```

---

### 6. Docker

Создан `Dockerfile` для сборки образа:

```dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Создан `docker-compose.yml` для запуска контейнера:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./experiments:/app/experiments
      - ./data:/app/data
    restart: unless-stopped
```

Команды для сборки и запуска:
```bash
docker build -t amazon-reviews-ml .
docker-compose up
```

Docker образ опубликован на DockerHub: `deg1t23/amazon-reviews-ml:latest`

---

### 7. Конфигурационные файлы

**config.ini** — гиперпараметры модели:
```ini
[model]
max_iter = 1000
C = 1.0
max_features = 10000
test_size = 0.2
random_state = 42
```

**requirements.txt** — зависимости:
```
scikit-learn==1.8.0
pandas==3.0.2
numpy==2.4.3
fastapi==0.136.0
uvicorn==0.46.0
pytest==7.4.0
joblib==1.5.3
pydantic==2.13.4
```

**dev_sec_ops.yml** — информация о сборке:
```yaml
docker_image:
  name: deg1t23/amazon-reviews-ml
  tag: latest

last_commits:
  - c8b8b068e8b3b8b12ad2c59a82c93fc5478993a3
  - c4d66c1fb070ab958824ccf18aaf33b33b7766b7
  - 7aaafed4df5b029fc13a35351422479145b40797
  - 7ae263f5f6ef86a383626d891ca2dd3cf5ea1d63
  - 488fea5126273230acdae52826c37f825f0dfe9e

test_coverage:
  total_tests: 5
  passed: 5
  failed: 0
```

---

### 8. CI Pipeline

CI pipeline реализован через **GitHub Actions** (`.github/workflows/ci.yml`).

**Триггер:** Pull Request в ветку `main`

**Шаги:**
1. Checkout кода
2. Установка Python 3.11
3. Установка зависимостей
4. Запуск тестов pytest
5. Авторизация в DockerHub
6. Сборка и публикация Docker образа

```yaml
name: CI Pipeline

on:
  pull_request:
    branches: [ main ]

jobs:
  test-and-build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -r requirements.txt httpx pytest

    - name: Run tests
      run: pytest tests/ -v

    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/amazon-reviews-ml:latest
```

**Результат:** CI pipeline успешно выполнен, образ опубликован на DockerHub.

---

### 9. CD Pipeline

CD pipeline реализован через **GitHub Actions** (`.github/workflows/cd.yml`).

**Триггер:** Завершение CI pipeline

**Шаги:**
1. Скачивание образа с DockerHub
2. Запуск контейнера
3. Ожидание запуска (10 секунд)
4. Функциональное тестирование эндпоинтов
5. Остановка контейнера

```yaml
name: CD Pipeline

on:
  workflow_run:
    workflows: ["CI Pipeline"]
    types:
      - completed
  workflow_dispatch:

jobs:
  deploy-and-test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Pull Docker image
      run: docker pull ${{ secrets.DOCKER_USERNAME }}/amazon-reviews-ml:latest

    - name: Run container
      run: |
        docker run -d --name test-container -p 8000:8000 \
        ${{ secrets.DOCKER_USERNAME }}/amazon-reviews-ml:latest

    - name: Wait for startup
      run: sleep 10

    - name: Test health endpoint
      run: curl -f http://localhost:8000/health

    - name: Test predict endpoint
      run: |
        curl -f -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "This album is absolutely amazing!"}'

    - name: Stop container
      run: docker stop test-container
```

**Результат:** CD pipeline успешно выполнен, функциональные тесты пройдены.

---

## Результаты работы

| Пункт | Статус |
|-------|--------|
| Репозиторий GitHub с историей коммитов | ✅ |
| Подготовка данных | ✅ |
| ML модель (accuracy 85%) | ✅ |
| Конвертация в .py + API сервис | ✅ |
| Тесты pytest (5/5) | ✅ |
| Docker образ | ✅ |
| config.ini | ✅ |
| Dockerfile + docker-compose.yml | ✅ |
| requirements.txt | ✅ |
| dev_sec_ops.yml | ✅ |
| CI pipeline (GitHub Actions) | ✅ |
| CD pipeline (GitHub Actions) | ✅ |
| Образ на DockerHub | ✅ |

---

## Выводы

В ходе лабораторной работы был реализован полный жизненный цикл разработки ML модели:

1. Разработана модель классификации тональности отзывов Amazon на основе логистической регрессии с точностью 85%
2. Модель обёрнута в REST API сервис на FastAPI
3. Написаны unit и интеграционные тесты
4. Приложение упаковано в Docker контейнер
5. Настроен CI/CD pipeline на GitHub Actions для автоматической сборки, тестирования и деплоя

Полученные навыки позволяют выстраивать production-ready ML инфраструктуру по стандартам MLOps.
