# Read me

All relevant files are in the `working` directory:

* `StreamFraud.ipynb` Python Notebook file with rendered graphs and tables detailing the entire throught processes and logic behind the analyses
* `fraud_detect_airflow.py` An apache airflow script that automates data retrieval from Google Cloud, data processing and analysis (using the `StreamFraud.py` script), and writes flagged user IDs to file (`flagged_user_id_timestamp.txt`)

**Note:** Due to lack of access, I could not write the final output to the orginal Google Cloud directory, and opted to write them locally instead.
