import joblib
import pandas as pd
import json

# This script simulates how the trained model would be used in production. It should have no training code — no fit, no GridSearchCV, no raw data loading from the API.

# Task 1: Load and Verify
# Load the Pipeline from models/weather_classifier.pkl. Load the metadata from models/weather_classifier_metadata.json and print the model's key metadata (city, features, test AUC).
clf = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test AUC: {metadata['test_auc']:.3f}\n")

# Task 2: Predict on New Data
# Create a DataFrame of at least five hypothetical days, covering a range of conditions: clearly good days, clearly bad days, and at least one borderline case. Use the feature names from your metadata to make sure the columns match exactly.
new_data = pd.DataFrame([
    {"temperature_2m_max": 20, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 10},  # clearly good
    {"temperature_2m_max": 5, "temperature_2m_min": -1, "precipitation_sum": 5.0, "wind_speed_10m_max": 35},   # clearly bad
    {"temperature_2m_max": 15, "temperature_2m_min": 5, "precipitation_sum": 1.0, "wind_speed_10m_max": 15},   # borderline good
    {"temperature_2m_max": 25, "temperature_2m_min": 18, "precipitation_sum": 0.0, "wind_speed_10m_max": 5},   # clearly good
    {"temperature_2m_max": 30, "temperature_2m_min": 20, "precipitation_sum": 0.0, "wind_speed_10m_max": 40},  # clearly bad
])
new_data = new_data[metadata['features']]

predictions = clf.predict(new_data)
probabilities = clf.predict_proba(new_data)[:, 1]  # Probability of being good for running

# For each day, print:
#
# The four input feature values
# The predicted label (good / skip)
# The model's confidence (probability of "good for running")
for i in range(len(new_data)):
    print(f"Day {i + 1}:")
    print(f"Features: {new_data.iloc[i].to_dict()}")
    print(f"Predicted label: {'good' if predictions[i] == 1 else 'skip'}")
    print(f"Model confidence (probability of 'good for running'): {probabilities[i]:.2f}\n")

# Task 3: Reflect
# Add a comment block answering the following:
#
# Pick the borderline case you included. What was the probability? Would you describe the model's answer as confident or uncertain? How would you handle a day where the model says 0.52?
# The training script and the prediction script are completely separate. What would break if someone ran predict_weather.py before train_weather_classifier.py? How would you make the error message more helpful?
# In a production system, the prediction script might run daily to classify tomorrow's weather forecast. What would need to change in predict_weather.py to support that? (You do not need to implement this — just describe it in a comment.)

"""
REFLECTION ANSWERS:
====================

1. Borderline Case Confidence & Handling:
   The borderline case (Day 3: temp_max=15°C, temp_min=5°C, precip=1.0mm, wind=15km/h) produces a probability around 0.50-0.60, 
   indicating modest uncertainty. The model is on the fence because this day is near the boundaries of our thresholds. 
   For a day with probability 0.52, I would flag it for human review or use a slightly higher decision threshold (like 0.55-0.60) 
   to avoid borderline recommendations. Alternatively, we could display a "try at your own risk" message instead of a binary good/skip.

2. Pre-Training Execution Risk & Error Handling:
   If predict_weather.py runs before train_weather_classifier.py is executed, the script would fail with a FileNotFoundError or 
   joblib.load/json.load error, which is poor UX. A better approach:
   - Add a try-except block that catches missing files and explains that the model must be trained first.
   - Store a model version or timestamp in metadata and check it on load.
   - Add a setup/validation step that checks for all required files before proceeding.

3. Production Daily Forecasting Changes:
   In production, predict_weather.py would need to:
   - Fetch tomorrow's weather forecast (from Open-Meteo forecast API instead of historical data)
   - Ensure feature order matches exactly (use metadata['features'] to validate column order)
   - Handle missing/invalid forecast data gracefully
   - Log predictions and confidence for monitoring
   - Possibly apply the adjusted threshold (e.g., 0.55) discussed above
   - Return results to a web API or send notifications to users
"""
