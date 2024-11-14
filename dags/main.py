from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    "astra", 
    start_date=datetime(2024, 11, 14), # datetime(2014, 7, 18), 
    schedule="0 12 * * *", # 8 am ET
    max_active_runs=1,
    catchup=False,
) as dag:

    task_migrate = BashOperator(
        task_id="migrate",
        bash_command="astra migrate"
    )


