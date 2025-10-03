from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

REPO_BRANCH = "dev"
APRED, RUN2D = ("1.5", "v6_2_1")

def skippy(*args, **kwargs):
    from airflow.exceptions import AirflowSkipException
    raise AirflowSkipException()


with DAG(
    "astra", 
    start_date=datetime(2024, 11, 14), # datetime(2014, 7, 18), 
    schedule="0 12 * * *", # 8 am ET
    max_active_runs=1,
    dagrun_timeout=timedelta(days=7),
    catchup=False,
) as dag:

    """
    repo = BashOperator(
        task_id="repo",
        bash_command="module load astra"
    )
    task_migrate = BashOperator(
        task_id="migrate",
        bash_command="sleep 1"
        #bash_command=f"astra migrate --apred {APRED}",# --run2d {RUN2D} --apred {APRED}",
    )
    """
    with TaskGroup(group_id="SummarySpectrumProducts") as summary_spectrum_products:
        BashOperator(task_id="mwmTargets", bash_command='astra create mwmTargets --overwrite')
        BashOperator(task_id="mwmAllVisit", bash_command='astra create mwmAllVisit --overwrite')
        BashOperator(task_id="mwmAllStar", bash_command='astra create mwmAllStar --overwrite')


    with TaskGroup(group_id="SpectrumProducts") as spectrum_products:
        (
            BashOperator(
                task_id="mwmVisit_mwmStar", 
                bash_command='astra srun astra.products.mwm.create_mwmVisit_and_mwmStar_products --nodes 10 --procs 4 --time="96:00:00"'
            )
        )

    with TaskGroup(group_id="ApogeeNet") as apogeenet:
        apogeenet_star = (
            BashOperator(
                task_id="star",
                bash_command='astra srun apogeenet apogee.ApogeeCoaddedSpectrumInApStar --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )
        )
        apogeenet_star >> (
            BashOperator(
                task_id="create_all_star_product",
                bash_command="astra create astraAllStarAPOGEENet --overwrite"
            )                
        )
        (
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
                #bash_command='astra srun aspcap --limit 10000 --nodes 8 --time="48:00:00"'
                #bash_command='astra srun aspcap --limit 125000 --nodes 10 --time="96:00:00"'
                bash_command='astra srun aspcap --limit 250000 --nodes 10 --time="96:00:00" --qos=sdss-np-urgent --partition=sdss-np --account=sdss-np'
            )
        ) >> (
            BashOperator(
                task_id="create_all_star_product",
                bash_command="astra create astraAllStarASPCAP --overwrite"
            )
        )
    
    with TaskGroup(group_id="BOSSNet") as bossnet:
        (
            BashOperator(
                task_id="star",
                bash_command='astra srun bossnet mwm.BossCombinedSpectrum --limit 250000 --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )

        ) >> (
            BashOperator(
                task_id="create_all_star_product",
                bash_command="astra create astraAllStarBOSSNet --overwrite"
            )
        )
        (
            BashOperator(
                task_id="visit",
                bash_command='astra srun bossnet boss.BossVisitSpectrum --limit 250000 --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="create_all_visit_product",
                bash_command="astra create astraAllVisitBOSSNet --overwrite"
            )
        )

    with TaskGroup(group_id="LineForest") as lineforest:
        (
            BashOperator(
                task_id="star",
                bash_command='astra srun line_forest mwm.BossCombinedSpectrum --limit 250000 --nodes 1 --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="create_all_star_product",
                bash_command="astra create astraAllStarLineForest --overwrite"
            )
        )
        (
            BashOperator(
                task_id="visit",
                bash_command='astra srun line_forest boss.BossVisitSpectrum --limit 250000 --nodes 1 --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="create_all_visit_product",
                bash_command="astra create astraAllVisitLineForest --overwrite"
            )
        )


    with TaskGroup(group_id="AstroNN") as astronn:
        astronn_star = BashOperator(
            task_id="star",
            bash_command='astra srun astronn --limit 500000 apogee.ApogeeCoaddedSpectrumInApStar --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
        )
        astronn_star >> (
            BashOperator(
                task_id="create_all_star_product",
                bash_command="astra create astraAllStarAstroNN --overwrite"
            )                
        )
        (
            BashOperator(
                task_id="visit",
                bash_command='astra srun astronn --limit 500000 apogee.ApogeeVisitSpectrumInApStar --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
            )
        ) >> (
            BashOperator(
                task_id="create_all_visit_product",
                bash_command="astra create astraAllVisitAstroNN --overwrite"
            )
        )

        with TaskGroup(group_id="AstroNN_Dist") as astronn_dist:
            (
                BashOperator(
                    task_id="astronn_dist",
                    bash_command='astra srun astronn_dist --mem=16000 --gres="gpu:v100" --account="notchpeak-gpu" --time="48:00:00"'
                )
            ) >> (
                BashOperator(
                    task_id="create_all_star_product",
                    bash_command="astra create astraAllStarAstroNNdist --overwrite"
                )
            )
        

        skipper = PythonOperator(
            task_id="skipper",
            python_callable=skippy,
        )
        astronn_star >> skipper >> astronn_dist

    summary_spectrum_products >> (apogeenet, aspcap, bossnet, lineforest, astronn)

    apogeenet_star >> aspcap >> spectrum_products

    #repo >> task_migrate

    #task_migrate >> (summary_spectrum_products, ) #spectrum_products)
    #task_migrate >> apogeenet
    #(task_migrate, apogeenet_star) >> aspcap
    #apogeenet_star >> aspcap

