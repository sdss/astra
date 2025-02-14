from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

APRED, RUN2D = ("1.4", "v6_2_0")

with DAG(
    "astra", 
    start_date=datetime(2024, 11, 14), # datetime(2014, 7, 18), 
    schedule="0 12 * * *", # 8 am ET
    max_active_runs=1,
    catchup=False,
) as dag:

    task_migrate = BashOperator(
        task_id="migrate",
        bash_command=f"astra migrate --apred {APRED}",# --run2d {RUN2D} --apred {APRED}",
        #execution_timeout=timedelta(hours=1)
    )

    with TaskGroup(group_id="SummarySpectrumProducts") as summary_spectrum_products:
        BashOperator(task_id="mwmTargets", bash_command='astra create mwmTargets --overwrite')
        BashOperator(task_id="mwmAllVisit", bash_command='astra create mwmAllVisit --overwrite')


    with TaskGroup(group_id="ApogeeNet") as apogeenet:
        (
            BashOperator(
                task_id="star",
                bash_command='astra srun apogeenet apogee.ApogeeCoaddedSpectrumInApStar --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="visit",
                bash_command='astra srun apogeenet apogee.ApogeeVisitSpectrumInApStar --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )
        )

    task_migrate >> summary_spectrum_products
    task_migrate >> apogeenet