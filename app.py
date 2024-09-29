from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl') 
scaler = joblib.load('scaler.pkl')  # Pre-trained scaler
pca = joblib.load('pca.pkl')  # Pre-trained PCA

app = FastAPI()

# Enable CORS for all origins (for development purposes)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Define input data structure
class InputFeatures(BaseModel):
    feature1: float = Field(..., description="Feature 1 description")
    feature2: float = Field(..., description="Feature 2 description")
    feature3: float = Field(..., description="Feature 3 description")
    feature4: float = Field(..., description="Feature 4 description")

# Simple HTML interface
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Brest Cancer Prediction</title>
        </head>
        <body>
            <h1>Brest Cancer Tumor Prediction</h1>
            <form action="/predict/" method="post">
                <label for="feature1">Feature 1:</label><br>
                <input type="text" id="feature1" name="feature1"><br>
                <label for="feature2">Feature 2:</label><br>
                <input type="text" id="feature2" name="feature2"><br>
                <label for="feature3">Feature 3:</label><br>
                <input type="text" id="feature3" name="feature3"><br>
                <label for="feature4">Feature 4:</label><br>
                <input type="text" id="feature4" name="feature4"><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict(features: InputFeatures):
    # Prepare input data
    input_data = np.array([[features.feature1, features.feature2, features.feature3, features.feature4]])  
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)

    # Make prediction
    prediction = model.predict(input_data_pca)
    predicted_class = "Cancer" if prediction[0] == 1 else "Not Cancer"

    return {"prediction": predicted_class}

# Error handling
@app.exception_handler(Exception)
async def validation_exception_handler(request, exc):
    return HTMLResponse(content=f"<html><body><h1>Error!</h1><p>{str(exc)}</p></body></html>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
