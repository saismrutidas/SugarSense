
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data in correct order
        features = [float(request.form[col]) for col in [
            'Glucose', 'BMI', 'Age', 'FamilyHistory', 'BloodPressure',
            'DailySugarIntake', 'PhysicalActivity', 'Gender', 'SmokingHistory', 'DrinkingHistory'
        ]]
        # Scale features
        scaled_features = scaler.transform([features])
        # Predict
        prediction = model.predict(scaled_features)[0]
        risk_level = 'High Risk' if prediction == 1 else 'Low Risk'
        recommendation = "Consult a healthcare provider and consider lifestyle changes, such as reducing sugar intake and increasing physical activity." if prediction == 1 else "Maintain a healthy lifestyle with balanced diet and regular exercise."
        return render_template('result.html', prediction_text=f"Diabetes Risk: {risk_level}", recommendation_text=f"Recommendation: {recommendation}")
    except Exception as e:
        return render_template('result.html', prediction_text="Error", recommendation_text="An error occurred: Please enter valid numerical values")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
