import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Task 1: Load and Explore
#
# The logistic regression lesson shows exactly how to load this dataset. Adapt that code for your script.
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

# I downloaded dataset cause url was causing an error
data_file = Path(__file__).resolve().parent / "spambase.data"
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(data_file, header=None)
df.columns = COLUMN_NAMES
print(df.head())
# Once it is loaded, take some time to understand what you are working with. How many emails are in the dataset?
print(f"Number of emails: {len(df)}")
print(f"Number of features: {len(df.columns) - 1}")
# How balanced are the two classes?
class_counts = df["spam_label"].value_counts()
print(class_counts)
spam_ratio = class_counts[1] / len(df) * 100
ham_ratio = class_counts[0] / len(df) * 100
print(f"Spam: {spam_ratio:.1f}%, Ham: {ham_ratio:.1f}%")
# What does that balance (or imbalance) mean for how you should interpret a raw accuracy score?
# With ~39% spam and ~61% ham a simple classifier predicting "ham" always will achieve ~61% accuracy
# But, raw accuracy without an additional info can be misleading
# We need precision, recall, and F1-score give a more complex picture

# Now explore how a few key features differ between spam and ham.
# For each of word_freq_free, char_freq_!, and capital_run_length_total, create a boxplot showing the distribution of that feature for spam emails versus ham emails.
# Save them to outputs/.
features_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    df.boxplot(column=feature, by="spam_label")
    plt.title(f"Boxplot of {feature} by Spam Label")
    plt.suptitle("")
    plt.xlabel("Spam Label (0 = Ham, 1 = Spam)")
    plt.ylabel(feature)
    plt.savefig(output_dir / f"{feature}_boxplot.png")
    plt.close()

# What do you notice? Are the differences between classes dramatic or subtle?
# I noticed that:
# - for word_freq_free spam emails tend to have higher values and more outliers
# - for char_freq_! spam emails have more ! chars
# - for capital_run_length_total spam often has much more ALL-CAPS usage
# There is a difference, but not a clear separation
# We need to combine features to get better results

# Then look at the raw scale of the features more broadly.
# Notice that many emails have a value of zero for most word-frequency features -- most emails do not contain the word "free" at all.
# What does this heavy skew toward zero tell you about the data?
# - The data is incomplete - many token-frequency features are absent in most emails
# - A few emails have large values giving long right tails and outliers

# Why does the numeric scale vary so dramatically across features (some are tiny fractions, others reach into the thousands)?
# - word_freq_* and char_freq_* are percentages/fractions and usually small
# - capital_run_length_* are count/run statistics and can be much larger
# - They have different meaning and units/ranges

# Why might that matter for some of the models you are about to build?
# It might matter for distance- and coefficient-based models (e.g. KNN, Logistic Regression, PCA) and we need to scale


# Task 2: Prepare Your Data
#
# Before building any models, prepare your data for the experiments in Task 3.
# You will need a train/test split and will need to think about how to handle the feature scales you noticed in Task 1.
X = df.drop(columns=["spam_label"])
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Document your choices in comments.
# Scale the data for KNN, Logistic Regression, and PCA.
# Fit scaler on training data only, then transform both train and test.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA preprocessing
# Not every classifier benefits from dimensionality reduction.
# Decision trees and random forests split on feature thresholds -- they are insensitive to feature scale or correlation,
# so PCA is unlikely to help them. KNN and logistic regression are different: both operate in a space where feature magnitudes matter and can benefit from reduced dimensionality.
# One rule applies whenever you use PCA: always scale the data first. PCA finds directions of maximum variance,
# so features with larger raw values will dominate unless you standardize first -- the same reason scaling is often used for KNN.
# For Spambase, where word frequencies are tiny fractions and capital_run_length_total can reach the thousands, this ordering is essential.
#
# Fit PCA on the training data only -- same reason as the scaler: fitting on all the data lets test-set information leak into the components.
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train_scaled)

cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
n = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(f"Number of components for >=90% variance: {n}")

# Plot the cumulative explained variance, save it to outputs/, and print n -- the number of components where it first reaches 90%.

plt.figure(figsize=(9, 6))
plt.plot(
    range(1, len(cumulative_explained_variance) + 1),
    cumulative_explained_variance,
    marker="o",
)
plt.axhline(y=0.90, color="red", linestyle="--", label="90% variance")
plt.axvline(x=n, color="green", linestyle="--", label=f"n={n}")
plt.title("PCA Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "pca_variance_explained.png")
plt.close()

# With n determined, transform both sets and slice to the first n components:
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

# Keep both the full scaled arrays and the PCA-reduced arrays -- you will use both in Task 3.

print(f"Scaled train shape: {X_train_scaled.shape}, Scaled test shape: {X_test_scaled.shape}")
print(f"PCA train shape: {X_train_pca.shape}, PCA test shape: {X_test_pca.shape}")

# Task 3: A Classifier Comparison
#
# Build and evaluate the following five classifiers. For each, print the accuracy and the full classification report.
#
# KNeighborsClassifier(n_neighbors=5) trained on the unscaled data
# KNeighborsClassifier(n_neighbors=5) trained on the scaled data, and again on the PCA-reduced data from Task 2 -- compare the two

print("\n--- 1. KNN with n_neighbors=5 (unscaled data) ---")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)
acc_knn_unscaled = accuracy_score(y_test, y_pred_knn_unscaled)
print(f"Accuracy: {acc_knn_unscaled:.4f}")
print(classification_report(y_test, y_pred_knn_unscaled, target_names=["Ham", "Spam"]))

print("\n--- 2. KNN with n_neighbors=5 (scaled data) ---")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
acc_knn_scaled = accuracy_score(y_test, y_pred_knn_scaled)
print(f"Accuracy: {acc_knn_scaled:.4f}")
print(classification_report(y_test, y_pred_knn_scaled, target_names=["Ham", "Spam"]))

print("\n--- 3. KNN with n_neighbors=5 (PCA-reduced data) ---")
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
acc_knn_pca = accuracy_score(y_test, y_pred_knn_pca)
print(f"Accuracy: {acc_knn_pca:.4f}")
print(classification_report(y_test, y_pred_knn_pca, target_names=["Ham", "Spam"]))

# DecisionTreeClassifier(random_state=42) -- before settling on a final depth, try max_depth values of 3, 5, 10, and None (unlimited).
# For each, print both the training accuracy and the test accuracy.
print("\n--- 4. DecisionTree: Depth Sweep (3, 5, 10, None) ---")
depths = [3, 5, 10, None]
dt_results = {}
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    dt_results[depth] = (train_acc, test_acc)
    print(f"max_depth={str(depth):>4}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# What do you notice as depth increases? What does that tell you about overfitting?
# Training accuracy keeps increasing with depth
# Test accuracy improves only for a little bit after depth 10
# The train-to-test gap grows a lot (`None` gap is biggest) which indicates probable overfitting risk

# Pick the depth you would use in production and add a comment explaining your reasoning.
# Test accuracy improves only a little after depth 10, so I decided on 10
chosen_depth = 10
# Then, using your chosen depth, print the accuracy and full classification report as you did for the other classifiers.
print(f"\nChosen depth for production: {chosen_depth}")
dt_final = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
dt_final.fit(X_train, y_train)
y_pred_dt = dt_final.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"Final DecisionTree Accuracy: {acc_dt:.4f}")
print(classification_report(y_test, y_pred_dt, target_names=["Ham", "Spam"]))

# RandomForestClassifier (introduced below)
print("\n--- 5. RandomForest ---")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=["Ham", "Spam"]))

# LogisticRegression(C=1.0, max_iter=1000, solver='liblinear') trained on the scaled data, and again on the PCA-reduced data -- compare the two
print("\n--- 6. LogisticRegression (scaled data) ---")
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_lr_scaled = lr_scaled.predict(X_test_scaled)
acc_lr_scaled = accuracy_score(y_test, y_pred_lr_scaled)
print(f"Accuracy: {acc_lr_scaled:.4f}")
print(classification_report(y_test, y_pred_lr_scaled, target_names=["Ham", "Spam"]))

print("\n--- 7. LogisticRegression (PCA-reduced data) ---")
lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
acc_lr_pca = accuracy_score(y_test, y_pred_lr_pca)
print(f"Accuracy: {acc_lr_pca:.4f}")
print(classification_report(y_test, y_pred_lr_pca, target_names=["Ham", "Spam"]))

# After you have results for all your classifiers, write a comment summarizing what you see. Which model performs best?
# For the classifiers where you compared PCA vs. non-PCA, which worked better -- and does that match your hypothesis from Task 2?
# For a spam filter specifically, is accuracy the right metric to optimize -- or would you rather minimize false positives (legitimate email marked as spam) or false negatives (spam that gets through)?
# Take a position and defend it.
print("Summary and analysis")
print("="*80)
print(f"KNN (unscaled):      {acc_knn_unscaled:.4f}")
print(f"KNN (scaled):        {acc_knn_scaled:.4f}")
print(f"KNN (PCA):           {acc_knn_pca:.4f}")
print(f"DecisionTree (d=10):  {acc_dt:.4f}")
print(f"RandomForest:        {acc_rf:.4f}")
print(f"LogisticReg (scaled):{acc_lr_scaled:.4f}")
print(f"LogisticReg (PCA):   {acc_lr_pca:.4f}")

# Among all classifiers RandomForest performed the best (accuracy 0.9457), with the second place by Logistic Regression on scaled features (0.9294)
# KNN improved dramatically after scaling (from 0.7991 to 0.9077), so distance-based models are highly scale-sensitive
# PCA did not improve performance for KNN or Logistic Regression, likely because the original features were already informative and PCA may have discarded some useful variance
# In the Decision Tree depths increased training accuracy a lot, but test accuracy improved only for a bit at high depth, so maybe some overfitting present

best_model_name = "RandomForest"
best_model = rf
best_predictions = y_pred_rf

print(f"\nBest model: {best_model_name} with accuracy {acc_rf:.4f}")

# For your best-performing classifier, create a confusion matrix using ConfusionMatrixDisplay and save it to outputs/best_model_confusion_matrix.png.
print("\nConfusion Matrix for Best Model:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.tight_layout()
plt.savefig(output_dir / "best_model_confusion_matrix.png")
plt.close()

# Given the costs described above, which type of error does your best model make more often?
# For spam filtering confusion matrix results show 18 false positives vs 32 false negatives
# Based on that the best model detects non-spam mails as spam relatively rarely, but still passes some spam through it

# Both the Decision Tree and the Random Forest expose a .feature_importances_ attribute.
# After building both, print the top 10 most important features for each and save a bar chart of the Random Forest importances to outputs/feature_importances.png.
def plot_feature_importances(importances, feature_names, model_name):
    indices = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), importances[indices], align="center")
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title(f"Top 10 Feature Importances for {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_feature_importances.png")
    plt.close()

dt_importances = dt_final.feature_importances_
rf_importances = rf.feature_importances_
feature_names = list(X_train.columns)

print("\nTop 10 Feature Importances - Decision Tree:")
dt_indices = np.argsort(dt_importances)[::-1][:10]
for i in dt_indices:
    print(f"  {feature_names[i]}: {dt_importances[i]:.4f}")

print("\nTop 10 Feature Importances - Random Forest:")
rf_indices = np.argsort(rf_importances)[::-1][:10]
for i in rf_indices:
    print(f"  {feature_names[i]}: {rf_importances[i]:.4f}")

plot_feature_importances(dt_importances, feature_names, "DecisionTree")
plot_feature_importances(rf_importances, feature_names, "RandomForest")

# Do the two models agree on which features matter most? Do the results match your intuition about what makes an email spam?
# Both models identify similar most-important features: char_freq_$, word_freq_remove, and char_freq_!
# But, their order is different: the Decision Tree overweights char_freq_$ (0.3887) a lot
# Random Forest orders importance more evenly across features (max 0.1145)
# This reflects their different learning styles:
# - Decision Trees can overfit to single discriminative features,
# - Random Forests order the decision-making across many features for robustness
# Classic spam indicators (!, $, "remove", "free") do appear in the top features, so they are close to my intuition


# Task 4: Cross-Validation
#
# A single train/test split can give you a misleading picture -- you might have gotten lucky (or unlucky) with how the data was divided.
# Cross-validation gives a more reliable estimate of how well a model generalizes to unseen data.
# Using cross_val_score with cv=5, run cross-validation on the training data for each of your classifiers from Task 3.
# For each, print the mean and standard deviation of the fold scores.
print("\n--- Cross-Validation Results ---")
models = {
    "KNN (unscaled)": KNeighborsClassifier(n_neighbors=5),
    "KNN (scaled)": KNeighborsClassifier(n_neighbors=5),
    "KNN (PCA)": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(max_depth=chosen_depth, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
    "LogisticReg (scaled)": LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42),
    "LogisticReg (PCA)": LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
}

cv_results = {}
for name, model in models.items():
    if "PCA" in name:
        X_cv = np.asarray(pca.transform(X_train_scaled)[:, :n])
    elif "scaled" in name:
        X_cv = np.asarray(X_train_scaled)
    else:
        X_cv = np.asarray(X_train)

    scores = cross_val_score(model, X_cv, y_train, cv=5, scoring="accuracy")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")

print("\nCross-Validation Summary:")
best_cv_model = max(cv_results.items(), key=lambda x: x[1][0])
most_stable_model = min(cv_results.items(), key=lambda x: x[1][1])
print(f"Most accurate (by CV): {best_cv_model[0]} (mean={best_cv_model[1][0]:.4f})")
print(f"Most stable (by CV): {most_stable_model[0]} (std={most_stable_model[1][1]:.4f})")

# Which model is the most accurate?
# Which is the most stable (lowest variance across folds)?
# Does the ranking match what you saw with the single train/test split?

# RandomForest is both the most accurate (mean 0.9543) and most stable (std 0.0133) across CV folds
# KNN (unscaled) now correctly shows 0.9046 accuracy, much lower than scaled (also 0.9046 in CV) despite the single test set showing 0.7991. This suggests the specific random split was unfavorable for unscaled KNN
# Logistic Regression (scaled) ranks second in stability, matching its strong single test performance (0.9294)
# The CV rankings align closely with single test rankings, which confirms RandomForest is the reliable choice


# Task 5: Building a Prediction Pipeline
#
# Build your pipelines
#
# Build two pipelines: one for your best tree-based classifier and one for your best non-tree-based classifier. For each, fit on the training data and print the full classification report on the test set. Confirm the results match your earlier manual approach. If your Task 3 experiments showed that PCA improved your non-tree model, include it as a step in that pipeline.
#
# Comment on your pipelines: do they have the same structure? Why or why not? What is the practical value of packaging a model this way, especially when handing it off to someone else or deploying it?

print("\n--- Task 5: Prediction Pipelines ---")

tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=100)),
])

if acc_lr_pca > acc_lr_scaled:
    non_tree_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)),
    ])
    non_tree_name = "LogisticRegression + Scaler + PCA"
else:
    non_tree_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)),
    ])
    non_tree_name = "LogisticRegression + Scaler"

tree_pipeline.fit(X_train, y_train)
tree_pipeline_pred = tree_pipeline.predict(X_test)
tree_pipeline_acc = accuracy_score(y_test, tree_pipeline_pred)

print("\nTree-based pipeline (RandomForest):")
print(f"Pipeline accuracy: {tree_pipeline_acc:.4f}")
print(classification_report(y_test, tree_pipeline_pred, target_names=["Ham", "Spam"]))
print(f"Manual RandomForest accuracy: {acc_rf:.4f}")

non_tree_pipeline.fit(X_train, y_train)
non_tree_pipeline_pred = non_tree_pipeline.predict(X_test)
non_tree_pipeline_acc = accuracy_score(y_test, non_tree_pipeline_pred)

print(f"\nNon-tree pipeline ({non_tree_name}):")
print(f"Pipeline accuracy: {non_tree_pipeline_acc:.4f}")
print(classification_report(y_test, non_tree_pipeline_pred, target_names=["Ham", "Spam"]))
print(f"Manual Logistic accuracy baseline: {max(acc_lr_scaled, acc_lr_pca):.4f}")

# Notes:
# - The tree pipeline has one step because RandomForest is scale-insensitive
# - The non-tree pipeline includes scaling (and PCA only if it helped in Task 3)
# - Pipelines reduce preprocessing mistakes and make deployment handoff safer
