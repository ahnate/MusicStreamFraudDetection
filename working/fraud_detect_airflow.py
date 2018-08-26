from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Ahnate',
    'depends_on_past': False,
    'start_date': datetime(2018, 8, 26),
    'email': ['ahnate@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('stream_fraud_detect', default_args=default_args)

# t1, t2 and t3 are examples of tasks created by instantiating operators
gsutil_command = """
    gsutil cp -r gs://1f59a9b2-e110-45a0-b2cf-02ec3d793098/* gs://music_fraud_detect/
    gsutil cp -r gs://music_fraud_detect/* /Users/ahnate/Files/DataScience/RecordUnionFraudDetection/input
"""

t1 = BashOperator(
    task_id='retrieve_files',
    bash_command=gsutil_command,
    dag=dag)

t2 = BashOperator(
    task_id='run_analysis',
    bash_command='python /Users/ahnate/Files/DataScience/RecordUnionFraudDetection/working/StreamFraud.py',
    dag=dag)

t3 = BashOperator(
    task_id='write_files',
    bash_command='gsutil cp /Users/ahnate/Files/DataScience/RecordUnionFraudDetection/working/*.txt gs://music_fraud_detect/',
    dag=dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
