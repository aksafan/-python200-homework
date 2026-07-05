import numpy as np
from sklearn import linear_model

# --- scikit-learn API ---
#
# scikit-learn Question 1
# The core pattern in scikit-learn is create → fit → predict. Practice it here with a simple dataset: years of work experience versus annual salary.
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
# Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.
model = linear_model.LinearRegression()
model.fit(years, salary)
new_years_4 = np.array([4]).reshape(-1, 1)
new_years_8 = np.array([8]).reshape(-1, 1)
prediction_4_years = model.predict(new_years_4)
prediction_8_years = model.predict(new_years_8)
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Predicted salary for 4 years of experience: {prediction_4_years[0]}")
print(f"Predicted salary for 8 years of experience: {prediction_8_years[0]}")

# scikit-learn Question 2
# scikit-learn requires the feature array X to be 2D even when you only have one feature. Start with this 1D array:
x = np.array([10, 20, 30, 40, 50])
print(f"Original shape: {x.shape}")
# Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D.
shaped_x = x.reshape(-1, 1)
print(f"Reshaped shape: {shaped_x.shape}")
# scikit-learn needs X to be 2D because it expects each row to show a sample and column to show a feature. In other way it will be an error. It should work the same way despite the number of elements/features

# scikit-learn Question 3
# K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn. Use the code below to generate a synthetic dataset with three natural clusters:

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Number of points in each cluster: {np.bincount(labels)}")

# Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png.
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
plt.title("Random Data Clusters Found by K-Means")
plt.xlabel("Random Data (synthetic scale)")
plt.ylabel("Random Data Y")
plt.tight_layout()
scatter_file = f'outputs/kmeans_clusters.png'
plt.savefig(scatter_file, dpi=150)
plt.close()


# --- Linear Regression ---

# The questions below all use the same synthetic medical costs dataset: 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), and an annual medical cost as the target. Generate it once and reuse the variables throughout.

import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Question 1
# Before fitting anything, look at the data.
# Create a scatter plot of age on the x-axis and cost on the y-axis.
# Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter().
# Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.tight_layout()
scatter_file = f'outputs/cost_vs_age.png'
plt.savefig(scatter_file, dpi=150)
plt.close()
# Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?
# Non-smokers definitely spend less money on medical. There two distinct slices as groups visible.

# Linear Regression Question 2
# Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.
age_reshaped = age.reshape(-1, 1)
age_train, age_test, cost_train, cost_test = train_test_split(
    age_reshaped, cost, test_size=0.2, random_state=42
)
print(f"age_train shape: {age_train.shape}")
print(f"age_test shape: {age_test.shape}")
print(f"cost_train shape: {cost_train.shape}")
print(f"cost_test shape: {cost_test.shape}")

# Linear Regression Question 3
# Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept.
model = LinearRegression()
model.fit(age_train, cost_train)
cost_pred = model.predict(age_test)
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Predicted costs for test set: {cost_pred}")
# Then predict on the test set and print:
rmse = np.sqrt(np.mean((cost_pred - cost_test) ** 2))
r2 = model.score(age_test, cost_test)
print(f"Root Mean Squared Error: {rmse}")
print(f"R² on test set: {r2}")
# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?
# The Slope: 196.57502840033254 means that for each additional year the increase in annual medical costs is around $196.58 - the older people will pay more
# Root Mean Squared Error: 6519.600511949668 means that an average difference between the predicted cost and the actual cost is about $6519.60
# R² on test set: 0.06951553917914943 means that the difference in costs can be explained by only age for about 6.95% - we need to count smoker status as well

# Linear Regression Question 4
# Now add smoker as a second feature and fit a new model.
X_full = np.column_stack([age, smoker])
# Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:
X_train_full, X_test_full, cost_train_full, cost_test_full = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train_full, cost_train_full)
cost_pred_full = model_full.predict(X_test_full)
r2_full = model_full.score(X_test_full, cost_test_full)
print(f"R² on test set with both features: {r2_full}")
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?
# Now it is much-much better - the R² on test set with both features is 0.7737232881262954 - means that the difference in costs can be explained by both age and smoker status for about 77.4%.
# The smoker coefficient: 14538.03793551 means that being a smoker adds about $14538.04 to the annual costs.

# Linear Regression Question 5
# A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point on the diagonal line where predicted equals actual.
# Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.
plt.scatter(cost_pred_full, cost_test_full, alpha=0.6, s=50)
min_val = min(cost_pred_full.min(), cost_test_full.min())
max_val = max(cost_pred_full.max(), cost_test_full.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/predicted_vs_actual.png', dpi=150)
plt.close()
# Add a comment: what does it mean when a point falls above the diagonal? What about below?
# When a point falls above the diagonal line it means that the model underestimated the actual cost (predicted < actual).
# When a point falls below the diagonal line it means that the model overestimated the actual cost (predicted > actual).
# Points on the diagonal should represent perfect predictions.
