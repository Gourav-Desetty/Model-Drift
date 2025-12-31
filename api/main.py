import os
import mlflow
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from models.image_classifier.image_model import ImageClassifier
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()
ml_models = {}

client = QdrantClient(
    url='https://cb31dd25-49e2-458f-a816-b7615fe710c7.us-east4-0.gcp.cloud.qdrant.io',
    api_key=os.getenv('QDRANT_API_KEY')
)

confidence_metrics = Gauge('model_confidence', 'Confidence score of the prediction', ['model_version'])

@asynccontextmanager
async def lifespan(app:FastAPI):
    ml_models['leukemia_clf'] = ImageClassifier(model_path="models/image_classifier/leukemia_model_densenet_121_01.pth")
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app)              # it automatically creates the /metric endpoint 

@app.post('/predict_image')
def predict_image(file: UploadFile=File(...)):
    # load the model
    model = ml_models['leukemia_clf']

    # make predictions
    result = model.predict(file.file)

    confidence_metrics.labels(model_version='densenet_121_v1').set(result['confidence'])

    file.file.seek(0)

    # load embeddings
    embeddings = model.embeddings(file.file)

    # save the predictions in database
    try:
        client.upsert(
            collection_name='model_drift',
            points=[
                PointStruct(
                    id=f"id_{str(uuid.uuid4())}",
                    vector=embeddings[0],
                    payload={
                        "model": 'densenet_121_v1',
                        'predictions': result['class'],
                        "Confidence": result['confidence']
                    }
                )
            ]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to log to Qdrant")

    # return back the predictions
    return JSONResponse(status_code=200, content=result)