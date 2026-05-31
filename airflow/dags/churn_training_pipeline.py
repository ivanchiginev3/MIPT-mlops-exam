from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="churn_training_pipeline",
    description="Training pipeline for customer churn prediction model",
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

    validate_model = BashOperator(
        task_id="validate_model",
        bash_command="test -f /app/models/churn_model.pkl",
    )

    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command="echo 'Model passed validation and is ready for deployment'",
    )

    validate_data >> train_model >> validate_model >> deploy_model