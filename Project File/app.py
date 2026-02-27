from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib


model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/index')
def index():
    return render_template("index.html")


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Start with all 70 features set to 0
    input_data = {feature: 0 for feature in feature_columns}

    # Only meaningful features user will fill
    user_features = [
        'funding_rounds',
        'funding_total_usd',
        'milestones',
        'relationships',
        'avg_participants',
        'age_first_funding_year',
        'age_last_funding_year',
        'age_first_milestone_year',
        'age_last_milestone_year',
        'is_top500'
    ]

    # Update only those provided by user
    for feature in user_features:
        value = request.form.get(feature)
        if value is not None and value != "":
            input_data[feature] = float(value)

    # Convert dictionary to DataFrame in correct order
    input_df = pd.DataFrame([input_data])[feature_columns]

    # Scale using trained scaler
    input_scaled = scaler.transform(input_df)

    # Predict

    prediction = model.predict(input_scaled)[0]

    return render_template("result.html", prediction=int(prediction))



if __name__ == "__main__":
    app.run(debug=True)
