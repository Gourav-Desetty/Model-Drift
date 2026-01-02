import os
import pandas as pd
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

load_dotenv()

client = QdrantClient(
    url='https://cb31dd25-49e2-458f-a816-b7615fe710c7.us-east4-0.gcp.cloud.qdrant.io',
    api_key=os.getenv('QDRANT_API_KEY')
)

points, _ = client.scroll(
    collection_name='model_drift',
    limit=10000
) 

if points is None:
    print("No records found go and make predictions")
    exit()

record = []
for point in points:
    payload = point.payload

    if not payload:
        continue

    if 'Confidence' in payload:
        payload['Confidence'] = float(payload['Confidence'])

    record.append(payload)

df = pd.DataFrame(record)

mid = len(df) // 2
ref_data = df[:mid]
curr_data = df[mid:]

report = Report(
    metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ]
)
report.run(reference_data=ref_data, current_data=curr_data)

report.save_html('drift_report.html')


report_dict = report.as_dict()

drift_score = -1

try:
    for metric in report_dict['metrics']:
        if metric['metric'] == 'DataDriftTable':
            drift_col = metric['result']['drift_by_columns']
            if 'Confidence' in drift_col:
                drift_score = float(drift_col['Confidence']['drift_score'] )
                break

    print(drift_score)

    if drift_score != -1:
        api_url = "http://localhost:8000/update_drift"
        try:
            payload = {'drift': drift_score }
            response = requests.post(api_url, json=payload)

            if response.status_code == 200:
                print("Sent to prometheus")
            else:
                print(f"API error: {response.text}")
        except Exception as e:
            print(f"Connection failed: Error {e}")
    else:
        print("Could not find confidence in DataDriftTable")
except Exception as e:
    print(f"Error: {e}")