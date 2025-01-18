from fastapi import FastAPI
from app.routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from fastapi import FastAPI, HTTPException, APIRouter
import pandas as pd
import numpy as np
from keras.models import load_model
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI(
    title="API for the DAS Homework",
    description="This api is is used to serve for the DAS Homework web application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://das-prototype.web.app"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (e.g., GET, POST)
    allow_headers=["*"],  # Allow all headers
)

logging.basicConfig(level=logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        body = await request.body()
        logging.info(f"Request body: {body.decode()}")
        response = await call_next(request)
        logging.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise e

model = load_model("app/lstm_model.h5")

class PredictionInput(BaseModel):
    input_data: dict

def preprocess_and_predict(input_data):
    input_data = input_data.drop(columns=['COMPANY', 'PRICE OF LAST TRANSACTION'])
    # Load the pre-trained model
    timesteps = model.input_shape[1]
    features = model.input_shape[2]

    # Ensure input_data has the correct number of features
    input_data = input_data.iloc[:, :features]

    # Handle missing values and normalize
    input_data = input_data.fillna(0)
    max_value = input_data.max().max()
    input_data_normalized = input_data / max_value

    # Check if there are enough rows for timesteps
    if len(input_data) < timesteps:
        raise ValueError(f"Input data must have at least {timesteps} rows for prediction.")

    # Reshape the data
    input_data_reshaped = np.array([input_data_normalized.values[-timesteps:]])
    input_data_reshaped = input_data_reshaped.reshape(1, timesteps, features)

    # Predict
    predictions = model.predict(input_data_reshaped)
    predictions_denormalized = predictions * max_value

    return predictions_denormalized.round()[0][0]


@app.get("/")
async def root():
    return {"message": "API for the DAS Homework"}

# API endpoint
@app.post("/predict/")
async def predict(payload: PredictionInput):
    try:
        input_data = payload.input_data
        dataframe = pd.DataFrame.from_dict(input_data)
        
        return {"prediction": preprocess_and_predict(input_data=dataframe)}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")