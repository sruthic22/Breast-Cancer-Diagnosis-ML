from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')  # Pre-trained model
scaler = joblib.load('scaler.pkl')  # Pre-trained scaler

app = FastAPI()

class InputFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict/")
def predict(features: InputFeatures):
    input_data = np.array([[features.feature1, features.feature2, features.feature3]]) 
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return {"prediction": "Cancer" if prediction[0] == 1 else "Not Cancer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
