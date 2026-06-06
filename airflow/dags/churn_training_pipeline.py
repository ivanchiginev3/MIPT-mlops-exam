from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="churn_training_pipeline",
    description="Customer churn MLOps pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "churn"],
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="test -f /app/data/Churn_Modelling.csv",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /app && python train.py",
    )

    validate_candidate = BashOperator(
        task_id="validate_candidate",
        bash_command="test -f /app/models/candidate/churn_model.pkl",
    )

    promote_model = BashOperator(
        task_id="promote_model",
        bash_command="cd /app && python promote_model.py",
    )

    validate_data >> train_model >> validate_candidate >> promote_model