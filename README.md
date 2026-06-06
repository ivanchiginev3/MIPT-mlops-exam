# Customer Churn Prediction MLOps System

## Overview

Данный проект представляет собой ML-систему для прогнозирования оттока клиентов банка.

Система включает полный жизненный цикл модели:

* подготовку данных;
* обучение модели;
* управление экспериментами через MLflow;
* инференс через FastAPI;
* контейнеризацию через Docker;
* оркестрацию через Airflow;
* CI/CD через GitHub Actions;
* мониторинг качества модели с использованием SLI/SLO;
* жизненный цикл модели через candidate, production и archive версии.

## Business Problem

Задача проекта — выявление клиентов с высоким риском оттока для последующего удержания.

Целевая переменная:

* `Exited = 1` — клиент ушел;
* `Exited = 0` — клиент остался.

## Model

Финальная модель:

**XGBoost Classifier**

Порог классификации:

```text
0.35
```

Метрики на тестовой выборке:

| Metric    | Value  |
| --------- | ------ |
| ROC-AUC   | 0.8661 |
| Precision | 0.6468 |
| Recall    | 0.6118 |
| F1-score  | 0.6288 |

## Model Lifecycle

В системе реализован упрощенный жизненный цикл модели:


Training
   ↓
Candidate model
   ↓
Quality evaluation
   ↓
Promotion
   ↓
Production model
   ↓
Archive previous production model


После запуска train.py новая модель сохраняется как candidate:
`models/candidate/churn_model.pkl`

После проверки качества запускается:
`python promote_model.py`

Скрипт выполняет promotion модели:
- текущая production-модель архивируется в models/archive/;
- candidate-модель копируется в models/production/;
- FastAPI использует только production-модель.

Если candidate-модель не соответствует SLO, promotion не выполняется, и production-сервис продолжает использовать предыдущую стабильную модель.

## Project Structure

```text
.
├── api/
│   └── main.py
├── airflow/
│   └── dags/
├── docs/
│   ├── manifesto.md
│   ├── sli_slo.md
│   └── adr_latency_decision.md
├── models/
│   ├── candidate/
│   ├── production/
│   └── archive/
├── notebooks/
│   └── mdd_analysis.ipynb
├── train.py
├── promote_model.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## API

### Health Check

```http
GET /health
```

### Prediction

```http
POST /predict
```

Пример запроса:

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 42,
  "Tenure": 2,
  "Balance": 50000,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 60000
}
```

## Local Run

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Обучение модели:

```bash
python train.py
python promote_model.py
```

Запуск API:

```bash
uvicorn api.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Docker

Сборка и запуск:

```bash
docker compose up -d --build
```

## MLflow

Запуск интерфейса:

```bash
mlflow ui
```

Доступ:

```text
http://127.0.0.1:5000
```

## Airflow Pipeline

Пайплайн обучения модели:

```text
validate_data
      ↓
train_model
      ↓
validate_model
      ↓
deploy_model
```

## Documentation

* `docs/manifesto.md`
* `docs/sli_slo.md`
* `docs/adr_latency_decision.md`

## Deployed Service

Yandex Cloud Serverless Container:

```text
https://bbavst4d7ls42rfp4cjl.containers.yandexcloud.net/
```
Health check:

```text
https://bbavst4d7ls42rfp4cjl.containers.yandexcloud.net/health
```

## Author

Ivan Chiginev

MIPT Data Science Master's Program
