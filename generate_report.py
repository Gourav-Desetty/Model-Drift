import os
import pandas as pd
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