from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel, conint, confloat

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the pre-trained model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Pydantic model (optional, for API validation if needed later)
class PredictionInput(BaseModel):
    Glucose: confloat(ge=0)
    BMI: confloat(ge=0)
    Age: conint(ge=0)
    FamilyHistory: conint(ge=0, le=1)
    BloodPressure: confloat(ge=0)
    DailySugarIntake: confloat(ge=0)
    PhysicalActivity: confloat(ge=0)
    Gender: conint(ge=0, le=1)
    SmokingHistory: conint(ge=0, le=1)
    DrinkingHistory: conint(ge=0, le=1)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Glucose: str = Form(...),  # Accept string input and convert manually
    BMI: str = Form(...),
    Age: str = Form(...),
    FamilyHistory: str = Form(...),
    BloodPressure: str = Form(...),
    DailySugarIntake: str = Form(...),
    PhysicalActivity: str = Form(...),
    Gender: str = Form(...),
    SmokingHistory: str = Form(...),
    DrinkingHistory: str = Form(...)
):
    try:
        # Convert string inputs to float/int where applicable
        features = [
            float(Glucose), float(BMI), float(Age), int(FamilyHistory),
            float(BloodPressure), float(DailySugarIntake), float(PhysicalActivity),
            int(Gender), int(SmokingHistory), int(DrinkingHistory)
        ]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        prediction_text = 'Diabetes Risk: High Risk' if prediction == 1 else 'Diabetes Risk: Low Risk'
        recommendation_text = 'Consult a doctor.' if prediction == 1 else 'Maintain a healthy lifestyle.'
    except (ValueError, TypeError):
        prediction_text = 'Error'
        recommendation_text = 'Please enter valid numerical inputs.'
    return templates.TemplateResponse("result.html", {"request": request, "prediction_text": prediction_text, "recommendation_text": recommendation_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)