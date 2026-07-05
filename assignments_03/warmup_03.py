# --- Preprocessing ---

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Preprocessing Question 1
# Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Print the shapes of all four arrays.
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Preprocessing Question 2
# Fit a StandardScaler on X_train and use it to transform both X_train and X_test.
std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)
# Print the mean of each column in X_train_scaled -- they should all be very close to 0.
print("Means of scaled features:", X_train_scaled.mean(axis=0))
# Add a comment explaining in one sentence why you fit the scaler on X_train only.
# I fit the scaler on X_train only to prevent data leakage, cause test data can influence the model

# --- KNN ---

# KNN Question 1
# Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), and predict on the test set. Print the accuracy score and the full classification report.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# KNN Question 2
# Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). Print the accuracy score. Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

preds_scaled = knn_scaled.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, preds_scaled))
print(classification_report(y_test, preds_scaled))
# Scaling improves KNN here because KNN uses distance, and standardization prevents larger-scale features from dominating nearest-neighbor comparisons

# KNN Question 3
# Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data.
knn_5 = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn_5, X_train, y_train, cv=5)
# Print each fold score, the mean, and the standard deviation.
print("k=5")
print("Fold scores:", scores)
print(f"Mean CV score: {scores.mean():.3f}")
print(f"Standard deviation of CV scores: {scores.std():.3f}")
# Add a comment: is this result more or less trustworthy than a single train/test split, and why?
# I think it is more trustworthy as it shows more realistic picture with 0.91 and 0.95 scores and standard deviation of 0.033
# It will help to see a real world situation, rather than relying on one potentially lucky or unlucky split

# KNN Question 4
# Loop over k values [1, 3, 5, 7, 9, 11, 13, 15].
# For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score.
for k in [1, 3, 5, 7, 9, 11, 13, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}")
#Add a comment identifying which k you would choose and why.
# My results are:
# k= 1:  mean=0.942
# k= 3:  mean=0.958
# k= 5:  mean=0.975
# k= 7:  mean=0.975
# k= 9:  mean=0.958
# k=11:  mean=0.958
# k=13:  mean=0.958
# k=15:  mean=0.967
# I'd choose k= 5:  mean=0.975 as it has the biggest mean value. Tho, k= 7:  mean=0.975 is also good, but I prefer smaller k cause it shows the first.


# --- Classifier Evaluation ---

# Classifier Evaluation Question 1
# Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay, passing display_labels=iris.target_names.
# Save the figure to outputs/knn_confusion_matrix.png. Add a comment: which pair of species does the model most often confuse (if any)?
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
disp.plot()
cm_file = output_dir / 'knn_confusion_matrix.png'
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig(cm_file, dpi=150)
plt.close()
# The model most often confuses versicolor and virginica, while setosa is classified almost perfectly

# --- The sklearn API: Decision Trees ---

# Decision Trees Question 1
# Create a DecisionTreeClassifier(max_depth=3, random_state=42), fit it on the unscaled training data, and predict on the test set.
dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
dtc.fit(X_train, y_train)
dtc_preds = dtc.predict(X_test)
# Print the accuracy score and classification report. Add a comment comparing the Decision Tree accuracy to KNN.
# Then add a second comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?
print("Decision Tree Accuracy:", accuracy_score(y_test, dtc_preds))
print(classification_report(y_test, dtc_preds))
# Decision Tree accuracy is 0.9666666666666667, which is the same as KNN accuracy on unscaled data
# Given that Decision Trees split data based on feature thresholds rather than distance calculations, scaling vs. unscaled data does not affect the result for Decision Trees.

# --- Logistic Regression and Regularization ---

# Logistic Regression Question 1
# Train three logistic regression models on the scaled Iris data, identical in every way except for the C parameter: C=0.01, C=1.0, and C=100.
# Use max_iter=1000 and solver='liblinear' for all three.
log_reg_0_1 = OneVsRestClassifier(LogisticRegression(C=0.01, max_iter=1000, solver="liblinear"))
log_reg_1_0 = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
log_reg_100 = OneVsRestClassifier(LogisticRegression(C=100, max_iter=1000, solver="liblinear"))
log_reg_0_1.fit(X_train_scaled, y_train)
log_reg_1_0.fit(X_train_scaled, y_train)
log_reg_100.fit(X_train_scaled, y_train)
# For each model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum().
coef_0_01 = np.abs(np.vstack([est.coef_ for est in log_reg_0_1.estimators_])).sum()
coef_1_0 = np.abs(np.vstack([est.coef_ for est in log_reg_1_0.estimators_])).sum()
coef_100 = np.abs(np.vstack([est.coef_ for est in log_reg_100.estimators_])).sum()
print("C=0.01, total coefficient magnitude:", coef_0_01)
print("C=1.0, total coefficient magnitude:", coef_1_0)
print("C=100, total coefficient magnitude:", coef_100)
# Add a comment: what happens to the total coefficient magnitude as C increases? What does this tell you about what regularization is doing?
# As C increases, the total coefficient magnitude grows (from 1.738029650758892 to 13.220936133674595 to 41.068106968999814
# C is the opposite of regularization: a small C means a strong regularization
# This shows that regularization is decreasing the model weights to reduce overfitting and improve generalization

# --- PCA ---

# The digits dataset is a collection of 1797 small handwritten digit images, each 8x8 pixels, bundled directly with scikit-learn (no download needed).
# Each image is stored as a flat array of 64 pixel values, so each sample lives in a 64-dimensional space -- a natural fit for dimensionality reduction.
# Pixel values range from 0 to 16, with higher values representing brighter pixels. The target labels are the digits 0 through 9.
# Add this data-loading block right before your PCA questions in warmup_03.py:
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# PCA Question 1
# Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9), using cmap='gray_r' with each digit's label as the title. Save the figure to outputs/sample_digits.png. (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker, so digits appear as dark ink on a light background, which is more readable than the default.)
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)
# Then create a 1-row subplot showing one example of each digit class (0-9), using cmap='gray_r' with each digit's label as the title.
# Save the figure to outputs/sample_digits.png. (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker, so digits appear as dark ink on a light background, which is more readable than the default.)
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for digit in range(10):
    # Find the first index of the current digit in y_digits
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(f"Digit {digit}")
    axes[digit].axis('off')
plt.tight_layout()
samp_digits_file = output_dir / 'sample_digits.png'
plt.savefig(samp_digits_file, dpi=150)
plt.close()

# PCA Question 2
# Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits).
pca = PCA(svd_solver="randomized", random_state=0)
pca.fit(X_digits)
scores = pca.transform(X_digits)
# As in the lesson, scores tell you how strongly each component is weighted for each sample -- scores[i, 0] is the weighting for PC1 in sample i, scores[i, 1] is the weighting for PC2, and so on.
# Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar.
# Here is the pattern for coloring by a label array and attaching a colorbar:
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2D Projection of Digits Dataset')
# Save the figure to outputs/pca_2d_projection.png. Add a comment: do same-digit images tend to cluster together in this 2D space?
plt.tight_layout()
samp_digits_file = output_dir / 'pca_2d_projection.png'
plt.savefig(samp_digits_file, dpi=150)
plt.close()
# Add a comment: do same-digit images tend to cluster together in this 2D space?
# Yes, same-digit images tend to cluster together in this 2D space
# This means that the PCA has captured some structures of the data that is different between the digit classes

# PCA Question 3
# Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components using np.cumsum(pca.explained_variance_ratio_).
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid()
# Save to outputs/pca_variance_explained.png.
plt.tight_layout()
samp_digits_file = output_dir / 'pca_variance_explained.png'
plt.savefig(samp_digits_file, dpi=150)
# Add a comment: approximately how many components do you need to explain 80% of the variance?
# I needed approximately 12 components to explain 80% of the variance, as we can see from the plot

# PCA Question 4
# The preprocessing lesson showed that a reconstruction is built by starting from the mean and adding each component weighted by its score.
# Here is the same idea generalized to n components -- add this function to your file:
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Using this function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits in X_digits using reconstruction through principal components n = 2, 5, 15, and 40.
n_values = [2, 5, 15, 40]
reconstructions = {n: [] for n in n_values}
for n in n_values:
    for i in range(5):
        recon = reconstruct_digit(i, scores, pca, n)
        reconstructions[n].append(recon)
# Build a grid of subplots where rows correspond to each n value and columns show those 5 digits.
fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(15, 10))
# Add an "Original" row at the top (use images[i], which is already shaped as (8, 8)).
for i in range(5):
    axes[0, i].imshow(images[i], cmap='gray_r')
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')
for row_idx, n in enumerate(n_values, start=1):
    for col_idx in range(5):
        axes[row_idx, col_idx].imshow(reconstructions[n][col_idx], cmap='gray_r')
        axes[row_idx, col_idx].set_title(f"n={n}")
        axes[row_idx, col_idx].axis('off')
plt.tight_layout()
# Save to outputs/pca_reconstructions.png.
samp_digits_file = output_dir / 'pca_reconstructions.png'
plt.savefig(samp_digits_file, dpi=150)
# Add a comment: at what n do the digits become clearly recognizable, and does that match where the variance curve levels off?
# At n=15, the digits become clearly recognizable
# It does match where the variance curve starts to level off around 80% explained variance
