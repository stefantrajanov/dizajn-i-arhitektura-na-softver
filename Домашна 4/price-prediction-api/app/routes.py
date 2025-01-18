from fastapi import FastAPI, HTTPException, APIRouter
import pandas as pd
import numpy as np
from keras.models import load_model
from pydantic import BaseModel

# Load the model
model = load_model("app/lstm_model.h5")
# Define the FastAPI app
router = APIRouter()

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

# API endpoint
@router.post("/predict/")
async def predict(payload: PredictionInput):
    try:
        input_data = payload.input_data
        dataframe = pd.DataFrame.from_dict(input_data)
        
        return {"prediction": preprocess_and_predict(input_data=dataframe)}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")