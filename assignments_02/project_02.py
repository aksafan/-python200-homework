# A separator is ";"

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Task 1: Load and Explore
# Load the dataset with the correct separator.
# Print the shape, the first five rows, and the data types of all columns.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_performance_math.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"

df = pd.read_csv(DATA_PATH, sep=";")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Then plot a histogram of G3 with 21 bins (one per possible value, 0-20).
# Add a title "Distribution of Final Math Grades", label both axes, and save to outputs/g3_distribution.png.
# You should see a cluster of zeros sitting apart from the main distribution.
# They represent the students who didn't take the final exam.
plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final grade (G3)")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "g3_distribution.png")
plt.close()

# Task 2: Preprocess the Data
# Handle the G3=0 rows first.
# Filter them out and save the result to a new DataFrame.
# Print the shape before and after to confirm how many rows were removed.
df_clean = df[df["G3"] > 0].copy()
print("Original shape:", df.shape)
print("Filtered shape:", df_clean.shape)

# Add a comment explaining your reasoning -- why would keeping these rows distort the model?
# Keeping these rows will distort results, cause lots of students have 0 on G3 (were absent), so it will impact as well

# Then convert the yes/no columns to 1/0 and the sex column to 0/1.
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for frame in (df, df_clean):
    frame[yes_no_cols] = frame[yes_no_cols].replace({"yes": 1, "no": 0})
    frame["sex"] = frame["sex"].map({"F": 0, "M": 1})

# Now check something interesting before moving on.
# Compute the Pearson correlation between absences and G3 on both the original dataset and the filtered one, and print both values.
# The difference is striking.
corr_original = df["absences"].corr(df["G3"])
corr_clean = df_clean["absences"].corr(df_clean["G3"])
print(f"Original data Pearson correlation: {corr_original:.3f}")
print(f"Filtered data Pearson correlation: {corr_clean:.3f}")

# Add a comment explaining why filtering changes the result: what were students with G3=0 doing in the original data that made absences look like a weak predictor?
# You might want to explore scatter plots to help understand this.

# Exploratory scatter plots: compare absences vs G3 before and after filtering G3 == 0.
plt.figure(figsize=(8, 5))
plt.scatter(df["absences"], df["G3"], alpha=0.6)
plt.title("Absences vs G3 (Original Data)")
plt.xlabel("Absences")
plt.ylabel("Final grade (G3)")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "absences_vs_g3_original.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df_clean["absences"], df_clean["G3"], alpha=0.6)
plt.title("Absences vs G3 (Filtered Data, G3 > 0)")
plt.xlabel("Absences")
plt.ylabel("Final grade (G3)")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "absences_vs_g3_filtered.png")
plt.close()

# Okay, now I see that having many G3=0 points on the original dataset changes the whole picture.
# Those G3=0 are from students who missed the final exam and doesn't show a proper state.
# It can be observed on generated scatter plots with a 0 at the bottom.#


# Task 3: Exploratory Data Analysis
# Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset,
# and print them sorted from most negative to most positive.
task3_numeric_feature_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
]
correlations = df_clean[task3_numeric_feature_cols].corrwith(df_clean["G3"]).sort_values()

print("\nTask 3 - Pearson correlations with G3 and sorted from most negative to most positive:")
print(correlations)

# Which feature has the strongest relationship with G3? Are any results surprising?
# The strongest relationship with G3 has:
# failures     -0.293831
# absences     -0.213129
# ...
# Fedu          0.158811
# Medu          0.190308
# The strongest relationship with G3 has failures (with -0.294) and absences (-0.213).
# The most surprising were parents' education (Fedu = 0.16, Medu = 0.19).

# Then create at least two visualizations of your own choosing and save them to outputs/.
# Use your judgment from previous weeks of data engineering to guide your use of plots.
# Use the correlation results to guide you -- what relationships seem worth a closer look?
# Add a comment for each plot describing what you see.

# Plot 1: boxplots that show the full grade distribution by each parent's education level
# It was interesting to me how each parents education impacted the final grades
medu_levels = sorted(df_clean["Medu"].unique())
fedu_levels = sorted(df_clean["Fedu"].unique())
medu_data = [df_clean.loc[df_clean["Medu"] == level, "G3"] for level in medu_levels]
fedu_data = [df_clean.loc[df_clean["Fedu"] == level, "G3"] for level in fedu_levels]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].boxplot(medu_data, tick_labels=medu_levels)
axes[0].set_title("G3 by Medu")
axes[0].set_xlabel("Mother's education (Medu)")
axes[0].set_ylabel("Final grade (G3)")
axes[1].boxplot(fedu_data, tick_labels=fedu_levels)
axes[1].set_title("G3 by Fedu")
axes[1].set_xlabel("Father's education (Fedu)")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "g3_by_parent_education_boxplots.png")
plt.close()

# Plot 2: average G3 and study time
# This one was obvious to me, but I decided to double-check on the distribution
# It still shows a positive trend: more study time - higher average G3
mean_g3_by_studytime = df_clean.groupby("studytime")["G3"].mean().sort_index()
plt.figure(figsize=(8, 5))
plt.plot(mean_g3_by_studytime.index, mean_g3_by_studytime.values, marker="o")
plt.title("Average Final Grade by Study Time")
plt.xlabel("Study time category (1-4)")
plt.ylabel("Average final grade (G3)")
plt.xticks([1, 2, 3, 4])
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "avg_g3_by_studytime.png")
plt.close()


# Task 4: Baseline Model
# Build the simplest possible model: use failures alone to predict G3.
# Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print the slope, RMSE, and R² on the test set.
X_baseline = df_clean[["failures"]].values
y_baseline = df_clean["G3"].values

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline,
    y_baseline,
    test_size=0.2,
    random_state=42,
)

baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)

y_pred_base = baseline_model.predict(X_test_base)
baseline_slope = baseline_model.coef_[0]
baseline_rmse = mean_squared_error(y_test_base, y_pred_base) ** 0.5
baseline_r2 = r2_score(y_test_base, y_pred_base)

print("\nTask 4 - Baseline model with failures and G3:")
print(f"Slope: {baseline_slope:+.3f}")
print(f"Test RMSE: {baseline_rmse:.3f}")
print(f"Test R^2: {baseline_r2:.3f}")

# Add a comment: given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English?
# Is R² better or worse than you expected from exploratory data analysis?
# For Slope: -1.428 each additional failure is associated with a drop of average 1.428 points for predicted G3
# For RMSE: 2.962 means that an average difference between the predicted G3 and the actual G3 is about 2.962 points which is quite high given the 0-20 scale
# For R^2: 0.089 means that the difference in G3 can be explained by only failures for about 8.9% - we need to count other features as well


# Task 5: Build the Full Model
# Now build a regression model using all of the numeric and binary features from the Feature Guide:

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

# Split into training and test sets (80/20, random_state=42), fit a LinearRegression model,
# and print both train R² and test R², as well as RMSE on the test set.
# Compare the test R² to your baseline from Task 4 -- how much does adding more features help?
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train_full, y_train_full)

y_pred_full = model.predict(X_test_full)
train_r2_full = model.score(X_train_full, y_train_full)
test_r2_full = model.score(X_test_full, y_test_full)
rmse_full = mean_squared_error(y_test_full, y_pred_full) ** 0.5

print("\nTask 5 - Full model:")
print(f"Train R^2: {train_r2_full:.3f}")
print(f"Test R^2: {test_r2_full:.3f}")
print(f"Test RMSE: {rmse_full:.3f}")
print(f"Baseline Test R^2 (Task 4): {baseline_r2:.3f}")
print(f"R^2 improvement over baseline: {test_r2_full - baseline_r2:.3f}")

# Print each feature name alongside its coefficient:
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Look carefully at the coefficients. Sort them mentally from largest to smallest.
# Are any signs (positive or negative) surprising given what you know about the data?
# For any surprising result, add a comment with your best explanation.
# Then compare train R² to test R² -- are they close, or is there a gap?
# What does that tell you about the model?

# The largest positive coefficients are for internet, higher, studytime, sex, Medu, Fedu, and the largest negative are for schoolsup and failures.
# The schoolsup (Extra educational support from the school (remedial help, tutoring)) was surprisingly with a negative coefficient.
# This is strange, cause I thought an additional support should be helpful.
# As for the positives they were expected, as students who have Internet and want to pursue higher education tend to perform better in their final grades.
# The positive signs for studytime, Medu, and Fedu are also expected, as more study time and higher parental education levels are usually associated with better academic performance.
# The Train R^2: 0.175 and the Test R^2: 0.154 are relatively close, so the model is not overfitting and should have had a reasonable generalization to unseen data.
# BUT the R² values are relatively small, so there might be other better features.

# Finally, add a comment answering: if you were deploying this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.
# For production, I'd keep features with a high coefficients that are relevant, e.g.: internet, studytime, higher, Medu/Fedu.


# Task 6: Evaluate and Summarize
# A useful way to evaluate a regression model visually is a predicted vs actual plot.
# This is a scatter plot where each point in the test set becomes a dot, with the model's prediction (y_hat) on the x-axis
# and the true value (y) on the y-axis.
# If the model were perfect, every point would fall exactly on the diagonal (predicted = actual).
# Clusters or curves away from the diagonal reveal systematic errors that RMSE alone won't show you.
# Random scattering around the diagonal is expected, and acceptable, prediction error.
# Create this plot for your test set. Add a diagonal reference line (for y=y_predicted), a title "Predicted vs Actual (Full Model)", labeled axes, and save to outputs/predicted_vs_actual.png.
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.7)
min_axis = min(y_pred_full.min(), y_test_full.min())
max_axis = max(y_pred_full.max(), y_test_full.max())
plt.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="red")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "predicted_vs_actual.png")
plt.close()

# Add a comment: does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels?
# What does a value above or below the diagonal mean?
low_band = y_test_full <= 8
high_band = y_test_full >= 14
low_band_mae = abs(y_test_full[low_band] - y_pred_full[low_band]).mean() if low_band.any() else float("nan")
high_band_mae = abs(y_test_full[high_band] - y_pred_full[high_band]).mean() if high_band.any() else float("nan")

# In this plot, a point above the diagonal means actual grade > predicted (the model under-predicted).
# A point below the diagonal means actual grade < predicted (the model over-predicted).
# Compare low/high bands by MAE to judge whether errors are roughly uniform or worse at one end.

# Task 6 - Error by grade band (MAE):
# Low-end MAE (actual <= 8): 3.892
# High-end MAE (actual >= 14): 3.023
# As we have low-end MAE bigger than high-end MAE, the model struggles more at the low end
# So, it seems to over-predict grades for the weakest students

# Then write a plain-language summary in your comments statements covering:
# The size of the filtered dataset and the test set
# The RMSE and R² of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?
# Which two features have the largest positive and largest negative coefficients, and what those mean
# One result that surprised you
coef_series = pd.Series(model.coef_, index=feature_cols)
coef_sorted_desc = coef_series.sort_values(ascending=False)
coef_sorted_asc = coef_series.sort_values()

# Task 6 - Plain-language summary:
# Filtered dataset size: 357 rows
# Test set size: 72 rows
# Best model RMSE is 2.855, so predictions are typically off by about 2.9 points on a 0-20 grade scale.
# Best model test R^2 is 0.154, so it explains about 15.4% of the variation in final grades.
# Largest positive coefficients: internet (+0.834) and higher (+0.610).
# Largest negative coefficients: schoolsup (-2.062) and failures (-1.145).

# One surprising result as I mentioned above is the strong negative coefficient for schoolsup
# A possible explanation might be that if students received support they may already be struggling a lot with their studying


# Neglected Feature: The Power of G1
# Add G1 (first period grade) as a feature to the full model from Task 5 and refit. We kept it out because it is so powerful. Print the new test R². The jump will be large -- from roughly 0.30 to somewhere around 0.80.
feature_cols_with_g1 = feature_cols + ["G1"]
X_with_g1 = df_clean[feature_cols_with_g1].values

a = train_test_split(
    X_with_g1,
    y,
    test_size=0.2,
    random_state=42,
)
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = a

model_with_g1 = LinearRegression()
model_with_g1.fit(X_train_g1, y_train_g1)

y_pred_g1 = model_with_g1.predict(X_test_g1)
test_r2_with_g1 = r2_score(y_test_g1, y_pred_g1)

# Neglected Feature - Power of G1:
# Test R^2 with G1 added: 0.749
# R^2 jump vs Task 5 model: 0.595

# Add a comment addressing these questions: does a high R² here mean G1 is causing G3?
# Is this a useful model for identifying students who might struggle?
# What might educators need to do if they wanted to intervene early, before G1 is even available?

# A high R^2 here does not mean a causation, cause G1 is mainly a very strong early indicator of later performance
# This model is useful for only short-term predictions and only after first-period grades are available
# For earlier intervention (before G1 is even available), schools need to think about other signals, e.g.: attendance patterns, study habits, support access, prior records, etc.
