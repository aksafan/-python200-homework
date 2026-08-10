import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report, f1_score,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- ROC and AUC ---

# ROC Question
#
# Train a LogisticRegression(max_iter=1000, random_state=42) on the raw (unscaled) training data and a KNeighborsClassifier(n_neighbors=5) on the scaled training data. For each model:
#
# Compute predicted probabilities on the test set using .predict_proba()
# Compute and print the AUC score using roc_auc_score
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_probs = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_probs)
print(f"Logistic Regression AUC: {auc:.4f}")
# Logistic Regression AUC: 0.7060

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
knn_auc = roc_auc_score(y_test, knn_probs)
print(f"KNeighborsClassifier AUC: {knn_auc:.3f}")
print(f"KNN (k=5) AUC:           {knn_auc:.3f}")
# KNeighborsClassifier AUC: 0.939
# KNN (k=5) AUC:           0.939

# Add a comment: which model has higher AUC? What does that tell you about which model better separates the two classes, independently of any threshold choice?
# The KNN model has a higher AUC than the Logistic Regression model.
# This indicates that the KNN model is better at separating the two classes across all possible threshold choices.
# A higher AUC means that the model has a better overall ability to discriminate between positive and negative classes, regardless of the specific threshold used for classification.

# ROC Question 2
#
# Plot both ROC curves on the same axes. Label each curve with the model name and its AUC score. Add the random-classifier diagonal. Save to outputs/roc_comparison.png.
fig, ax = plt.subplots(figsize=(6, 5))
log_fpr, log_tpr, thresholds = roc_curve(y_test, y_probs)
RocCurveDisplay(fpr=log_fpr, tpr=log_tpr).plot(ax=ax, name=f"Logistic Regression (AUC={auc:.2f})")
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
RocCurveDisplay(fpr=knn_fpr, tpr=knn_tpr).plot(ax=ax, name=f"KNN k=5 (AUC={knn_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("ROC Comparison — Weather Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.show()

# Add a comment: at the point on each curve where TPR = 0.80, which model has the lower FPR? What does that mean practically — if you needed to catch 80% of positives, which model would produce fewer false alarms?
# At the point where TPR = 0.80, KNN model has the lower FPR (~0.05).
# This means that if we need to catch 80% of positives, the KNN model would produce fewer false alarms (false positives) compared to the Logistic Regression model.
# In practical terms, this would make the KNN model more desirable in scenarios where minimizing false positives is important while still maintaining a high true positive rate.

# ROC Question 3
#
# Using the logistic regression from Q1, find the threshold that achieves the highest F1 score on the test set. To do this:
#
# Get fpr, tpr, and thresholds from roc_curve(y_test, y_probs_lr).
# For each threshold, compute y_pred = (y_probs_lr >= threshold).astype(int) and calculate the F1 score.
f1_scores = []
for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# Find the index of the best F1 score
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_tpr = log_tpr[best_idx]
best_fpr = log_fpr[best_idx]

# Print the threshold, TPR, FPR, and F1 at the optimum
print(f"Best Threshold: {best_threshold:.3f}, TPR: {best_tpr:.3f}, FPR: {best_fpr:.3f}, F1: {best_f1:.3f}")

# Add a comment: how does this optimal threshold compare to the default 0.5? In a real application, when would you choose a threshold lower than 0.5?
# The optimal threshold for the highest F1 score is lower than the default threshold of 0.5.
# In a real application, you would choose a threshold lower than 0.5 when you want to increase the sensitivity (true positive rate) of the model, even if it means accepting more false positives.
# This is particularly important in scenarios where missing a positive case has serious consequences, such as in medical diagnoses or fraud detection.


# --- GridSearchCV ---
#
# GridSearch Question 1
# Build a Pipeline with a StandardScaler and a LogisticRegression(max_iter=1000). Use GridSearchCV with cv=5 and scoring="roc_auc" to search over C values [0.001, 0.01, 0.1, 1.0, 10.0, 100.0].
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])
param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_pipe = grid_search.best_estimator_
y_probs = best_pipe.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs)
# Print:
#
# The best C value
# The best CV AUC score
# The test AUC of the best estimator
print(f"Best C: {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
print(f"Test AUC: {test_auc:.3f}")
# Best C: 100.0
# Best CV AUC: 0.773
# Test AUC: 0.706
# Add a comment: did the grid search pick the same C you would have guessed by default? By how much did the test AUC change compared to the default C=1.0?
# Grid search did *not* pick the default C=1.0; it selected C=100.0.
# The test AUC changed from 0.7060 to 0.706, so the change is approximately 0.000 (effectively no meaningful improvement).

# GridSearch Question 2
# Run a second grid search using the same Pipeline, but this time replace the LogisticRegression with a DecisionTreeClassifier(random_state=42) and search over max_depth values [2, 3, 5, 8, None].
pipe_dtc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    DecisionTreeClassifier(random_state=42)),
])
param_grid_dtc = {"clf__max_depth": [2, 3, 5, 8, None]}

grid_search_dtc = GridSearchCV(pipe_dtc, param_grid_dtc, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search_dtc.fit(X_train, y_train)

best_pipe_dtc = grid_search_dtc.best_estimator_
y_probs_dtc = best_pipe_dtc.predict_proba(X_test)[:, 1]
test_auc_dtc = roc_auc_score(y_test, y_probs_dtc)
# Print the best max_depth and best CV AUC, then print the test AUC.
print(f"Best DTC max_depth: {grid_search_dtc.best_params_['clf__max_depth']}")
print(f"Best DTC CV AUC: {grid_search_dtc.best_score_:.3f}")
print(f"Test DTC AUC: {test_auc_dtc:.3f}")
# Best DTC max_depth: 5
# Best DTC CV AUC: 0.917
# Test DTC AUC: 0.935
# Add a comment: compare the best AUC from Q1 (logistic regression) to this one (decision tree). Which model would you bring into further development? Is AUC the only thing you would consider?
# The best AUC from Q1 (logistic regression) is 0.773, while the best AUC from Q2 (decision tree) is 0.917.
# The decision tree model has a significantly higher AUC, indicating better performance in separating the classes.
# I would bring the decision tree model into further development due to its superior AUC.
# However, AUC is not the only consideration; factors like model interpretability, complexity, training time, and overfitting potential should also be evaluated before making a final decision.

# GridSearch Question 3
# Look at the cv_results_ from either grid search. Print the mean and standard deviation of the CV AUC for each parameter value, sorted from best to worst.
cv_results = grid_search.cv_results_
mean_auc = cv_results["mean_test_score"]
std_auc = cv_results["std_test_score"]
params = cv_results["params"]

sorted_indices = np.argsort(mean_auc)[::-1]
print("Logistic Regression CV Results:")
for idx in sorted_indices:
    print(f"Params: {params[idx]}, Mean AUC: {mean_auc[idx]:.3f}, Std AUC: {std_auc[idx]:.3f}")
# Decision Tree CV Results:
# Params: {'clf__max_depth': 5}, Mean AUC: 0.917, Std AUC: 0.021
# Params: {'clf__max_depth': 3}, Mean AUC: 0.902, Std AUC: 0.019
# Params: {'clf__max_depth': 8}, Mean AUC: 0.881, Std AUC: 0.026
# Params: {'clf__max_depth': None}, Mean AUC: 0.863, Std AUC: 0.039
# Params: {'clf__max_depth': 2}, Mean AUC: 0.823, Std AUC: 0.015

cv_results_dtc = grid_search_dtc.cv_results_
mean_auc_dtc = cv_results_dtc["mean_test_score"]
std_auc_dtc = cv_results_dtc["std_test_score"]
params_dtc = cv_results_dtc["params"]

sorted_indices_dtc = np.argsort(mean_auc_dtc)[::-1]
print("\nDecision Tree CV Results:")
for idx in sorted_indices_dtc:
    print(f"Params: {params_dtc[idx]}, Mean AUC: {mean_auc_dtc[idx]:.3f}, Std AUC: {std_auc_dtc[idx]:.3f}")
# Logistic Regression CV Results:
# Params: {'clf__C': 100.0}, Mean AUC: 0.773, Std AUC: 0.006
# Params: {'clf__C': 10.0}, Mean AUC: 0.773, Std AUC: 0.006
# Params: {'clf__C': 1.0}, Mean AUC: 0.772, Std AUC: 0.006
# Params: {'clf__C': 0.1}, Mean AUC: 0.772, Std AUC: 0.007
# Params: {'clf__C': 0.01}, Mean AUC: 0.762, Std AUC: 0.014
# Params: {'clf__C': 0.001}, Mean AUC: 0.735, Std AUC: 0.022

# Add a comment: find a case where two parameter values have similar mean scores but different standard deviations. If you had to choose between them, which would you pick and why?
# For the Logistic Regression CV Results, C=10.0 and C=0.1 have similar mean AUC scores (0.773 vs 0.772) but the C=0.1 has a slightly higher standard deviation (0.007 vs 0.006).
# If I had to choose between them, I would pick C=10.0 because it has a lower standard deviation (0.006), indicating more consistent performance across the cross-validation folds.
# A lower standard deviation suggests that the model's performance is more stable and less sensitive to variations in the training data, making it more reliable in production.


# --- joblib ---

# joblib Question 1
# Take the best Pipeline from GridSearch Question 1 (the logistic regression pipeline). Save it to models/warmup_model.pkl using joblib.dump. Then load it back in the same script using joblib.load and confirm it makes identical predictions to the original:
# Note: best_pipe already exists from GridSearch Question 1, so we reuse it here
joblib.dump(best_pipe, "models/warmup_model.pkl")

loaded_clf = joblib.load("models/warmup_model.pkl")
original_preds = best_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)
assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")
# Add a comment: what would break if you saved only the logistic regression model (without the scaler) and then called .predict(X_test) on the loaded model, where X_test is unscaled?
# If I save only the logistic regression model without the scaler, the predictions would likely be incorrect because the model was trained on scaled data. The features in X_test would not be scaled, leading to inconsistent input ranges and potentially poor performance.

# joblib Question 2
# Demonstrate a minimal version of the train/predict split. In the same warmup_04.py:
#
# Save the best logistic regression Pipeline to models/warmup_model.pkl (you may have already done this in Q1).
# Add a clearly labeled section — use a comment like # --- Simulated prediction script ---.
# In that section, load the model fresh from disk and use it to predict on three manually constructed rows:
#
import numpy as np

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

# --- Simulated prediction script ---
loaded_clf_fresh = joblib.load("models/warmup_model.pkl")

for i, sample in enumerate(new_samples):
    pred_class = loaded_clf_fresh.predict([sample])[0]
    pred_prob = loaded_clf_fresh.predict_proba([sample])[0, 1]
    print(f"Sample {i+1}: Predicted class = {pred_class}, Probability = {pred_prob:.3f}")
# Sample 1: Predicted class = 1, Probability = 0.750
# Sample 2: Predicted class = 1, Probability = 0.743
# Sample 3: Predicted class = 1, Probability = 0.653

# Print the predicted class and the probability for each row. Add a comment: what do you expect the all-zeros row to predict? Why?
# I expect the all-zeros row to predict class 1 with probability ~0.65 because, after scaling, zeros represent the mean of the feature distribution.
# The trained model learned that samples near the mean (zeros in scaled space) have a higher likelihood of being class 1, hence the prediction of class 1 with probability 0.653.
