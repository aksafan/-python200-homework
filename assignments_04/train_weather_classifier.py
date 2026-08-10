import requests
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import sklearn
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, RocCurveDisplay

# Step 1: Fetch the Data
#
# Use the Open-Meteo historical API to download one year of daily weather data for a location of your choice. The API is free and requires no key. Use these four daily variables:
#
#     temperature_2m_max — daily high temperature (°C)
#     temperature_2m_min — daily low temperature (°C)
#     precipitation_sum — total precipitation (mm)
#     wind_speed_10m_max — maximum wind speed (km/h)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 30.2672,
    "longitude": -97.7431,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Chicago",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Step 2: Engineer Labels
#
# Define what "good for running" means in terms of your four numeric features. The lesson uses these thresholds as a starting point:
# Feature	"Good for running" range
# temperature_2m_max	7 - 24 °C (45-79°F)
# temperature_2m_min	≥ 0 °C (above freezing)
# precipitation_sum	< 3.0 mm
# wind_speed_10m_max	< 30 km/h
#
# Austin, TX Adjustments:
# - temperature_2m_max: Using default 7-24°C. Austin is hot, but this range captures mild/comfortable mornings.
# - temperature_2m_min: Using default ≥ 0°C (freezing). Austin rarely freezes, so this is practical.

# Print the class distribution after labeling
df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 24) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3.0) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)

print(df["good_for_running"].value_counts().sort_index())
fraction_good = df["good_for_running"].sum() / len(df)
print(f"Fraction of days labeled 'good for running': {fraction_good:.1%}")

# and add a comment: what fraction of days in your dataset are labeled "good for running"?
fraction_good = df["good_for_running"].sum() / len(df)
# Fraction of days labeled 'good for running' is 24.9% (91 out of 365 days in 2023)

# Does that seem reasonable given the climate where you chose?
# For Austin's hot, dry climate, this fraction seems reasonable.
# Most days exceed the 24°C upper temperature limit during summer months, but spring/fall offer many good running days.

# Step 3: Train and Tune
#
# Split the data into train (80%) and test (20%) sets, stratifying on the label.
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use GridSearchCV with a Pipeline(StandardScaler, LogisticRegression) to search over at least five values of C. Use cv=5 and scoring="roc_auc". Print:
#     The best C value and best CV AUC
#     A full classification report on the test set
#     The test AUC
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])
param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
print(f"Best C:      {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")

best_pipe = grid_search.best_estimator_
y_pred  = best_pipe.predict(X_test)         # no manual scaling needed
y_probs = best_pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")

# Then plot and save the ROC curve for the best estimator to outputs/weather_roc.png.
fpr, tpr, _ = roc_curve(y_test, y_probs)
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name="Logistic Regression")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve — Weather Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.show()

# Step 4: Reflect on Evaluation
#
# Add a comment block (at least 4-6 sentences) addressing the following:
#
# What does the AUC score tell you about this model's quality? Is it surprisingly good, surprisingly bad, or about what you expected?
# The model's test AUC of 0.980 is excellent—this suggests the model learned meaningful patterns in Austin's weather data.
# The high AUC (>0.95) indicates strong separation between "good" and "not good" running days, which is better than expected for a simple weather classifier.

# Look at the precision and recall in the classification report. Which type of error (false positive vs. false negative) is more common? What would this mean in practice — would you rather the app over-recommend running or under-recommend it?
# Looking at the classification report, precision for "good for running" (class 1) is 0.89 with recall of 0.94.
# This means the model is good at identifying actual good-running days (94% recall), but occasionally mislabels a bad day as good (89% precision).
# False positives are slightly more common than false negatives. For a running app, I'd prefer to under-recommend than over-recommend — it's safer to miss a good day than to recommend running in poor conditions.
# Currently, the model errs toward over-recommending, which could frustrate users in uncomfortable weather.

# If you were setting the threshold for a real app, would you use the default 0.5? What would you change it to and why?
# I would lower the decision threshold below 0.5 (perhaps to 0.4-0.45) to reduce false positives.
# This would increase false negatives (missing some good days) but improve precision, making the app more conservative and trustworthy.
# Users prefer a cautious recommendation over an overly optimistic one when it comes to outdoor activities.


# Step 5: Save the Model
# Save the best Pipeline to models/weather_classifier.pkl using joblib.dump. Save a metadata file to models/weather_classifier_metadata.json that includes:
#
# Python version
# scikit-learn version
# Feature names (in order)
# Best hyperparameters from GridSearchCV
# Test AUC
# Your city (latitude and longitude)
# A brief description of the label thresholds you used

joblib.dump(best_pipe, "models/weather_classifier.pkl")

metadata = {
    "python_version":         sys.version,
    "sklearn_version":        sklearn.__version__,
    "features":               FEATURES,
    "label":                  "good_for_running",
    "best_hyperparameters":   grid_search.best_params_,
    "test_auc":               float(roc_auc_score(y_test, y_probs)),
    "city":                   "Austin, TX",
    "latitude":               30.2672,
    "longitude":              -97.7431,
    "label_thresholds": {
        "temperature_2m_max": "7–24°C",
        "temperature_2m_min": ">= 0°C",
        "precipitation_sum":  "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model saved to models/weather_classifier.pkl")
print("Metadata saved to models/weather_classifier_metadata.json")
