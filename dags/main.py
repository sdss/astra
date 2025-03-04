from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

REPO_BRANCH = "dev"
APRED, RUN2D = ("1.4", "v6_2_0")

# APRED for DR20 will be 1.5
# RUN2D for DR20 will be v6_2_0


with DAG(
    "astra", 
    start_date=datetime(2024, 11, 14), # datetime(2014, 7, 18), 
    schedule="0 12 * * *", # 8 am ET
    max_active_runs=1,
    catchup=False,
) as dag:

    repo = BashOperator(
        task_id="repo",
        bash_command=(
            f"cd $MWM_ASTRA/software/astra; "
            f"git checkout {REPO_BRANCH}; "
            f"git pull"
        ),
    )

    """
    task_migrate = BashOperator(
        task_id="migrate",
        bash_command=f"astra migrate --apred {APRED}",# --run2d {RUN2D} --apred {APRED}",
    )
    """
    task_migrate = BashOperator(
        task_id="migrate",
        bash_command="sleep 1"
    )
    
    with TaskGroup(group_id="SummarySpectrumProducts") as summary_spectrum_products:
        BashOperator(task_id="mwmTargets", bash_command='astra create mwmTargets --overwrite')
        BashOperator(task_id="mwmAllVisit", bash_command='astra create mwmAllVisit --overwrite')

    with TaskGroup(group_id="SpectrumProducts") as spectrum_products:
        BashOperator(task_id="mwmVisit_mwmStar", bash_command="astra srun astra.products.mwm.create_mwmVisit_and_mwmStar_products --nodes 1")

    with TaskGroup(group_id="ApogeeNet") as apogeenet:
        (
            BashOperator(
                task_id="star",
                bash_command='astra srun apogeenet apogee.ApogeeCoaddedSpectrumInApStar --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="create_all_star_product",
                bash_command="astra create astraAllStarAPOGEENet --overwrite"
            )                
        ) >> (
            BashOperator(
                task_id="visit",
                bash_command='astra srun apogeenet apogee.ApogeeVisitSpectrumInApStar --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="create_all_visit_product",
                bash_command="astra create astraAllVisitAPOGEENet --overwrite"
            )
        )

    with TaskGroup(group_id="ASPCAP") as aspcap:
        (
            BashOperator(
                task_id="aspcap",
                # We should be able to do ~20,000 spectra per node per day.
                # To be safe while testing, let's do 4 nodes with 40,000 spectra (should be approx 12 hrs wall time)
                bash_command='astra srun aspcap --limit 100000 --nodes 8 --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="create_all_star_product",
                bash_command="astra create astraAllStarASPCAP --overwrite"
            )
        )
            


    repo >> task_migrate

    task_migrate >> (summary_spectrum_products, spectrum_products)
    task_migrate >> apogeenet
    task_migrate >> aspcap
