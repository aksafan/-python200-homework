import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# --- Pandas ---

# Pandas Question 1
# Create the following DataFrame and print the first three rows, the shape, and the data types of each column.
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"Num Rows: {len(df)}")
print(f"First three rows: \n{df.head(3)}")
print(f"Shape: {df.shape}")
print(f"Data types: \n{df.dtypes}")

# Pandas Question 2
# Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.
print(f"Students who passed and have a grade above 80: \n {df.where((df['passed'] == True) & (df['grade'] > 80))}")
print(f"Students who passed and have a grade above 80 (no NaN): \n {df.where((df['passed'] == True) & (df['grade'] > 80)).dropna()}")

# Pandas Question 3
# Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).
df['grade_curved'] = df['grade'] + 5
print(f"DataFrame with 'grade_curved': \n {df}")

# Pandas Question 4
# Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.
df['name_upper'] = df['name'].str.upper()
print(f"DataFrame with 'name' and 'name_upper' columns together: \n {df[['name', 'name_upper']]}")

# Pandas Question 5
# Group the DataFrame by "city" and compute the mean grade for each city. Print the result.
df_grouped_by_city = df.groupby('city')['grade'].mean()
print(f"Mean grade for each city: \n {df_grouped_by_city}")

# Pandas Question 6
# Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
df['city'] = df['city'].str.replace('Austin', 'Houston')
print(f"DF with 'name' and 'city' columns after replacement: \n {df[['name', 'city']]}")

# Pandas Question 7
# Sort the DataFrame by "grade" in descending order and print the top 3 rows.
df_sorted = df.sort_values('grade', ascending=False)
print(f"DF sorted by 'grade' in descending order (top 3 rows): \n {df_sorted.head(3)}")


# --- NumPy ---

# NumPy Question 1
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.
arr = np.array([10, 20, 30, 40, 50])
print(f"1D NumPy array. Shape: {arr.shape}, Dtype: {arr.dtype}, Ndim: {arr.ndim}")

# NumPy Question 2
# Create the following 2D array and print its shape and size (total number of elements).
arr2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"2D NumPy array. Shape: {arr2.shape}, Size: {arr2.size}")

# NumPy Question 3
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].
arr2_sliced = arr2[:2, :2]
print(f"Sliced 2x2 block from the top-left of the array: \n {arr2_sliced}")

# NumPy Question 4
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.
arr3_4 = np.zeros(3*4)
arr2_5 = np.ones(2*5)
print(f"3x4 array of zeros: \n {arr3_4.reshape(3, 4)}")
print(f"2x5 array of ones: \n {arr2_5.reshape(2, 5)}")

# NumPy Question 5
# Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.
arr_from_range = np.arange(0, 50, 5)
print(f"Array from np.arange(0, 50, 5): \n {arr_from_range}")
print(f"Shape: {arr_from_range.shape}, Mean: {arr_from_range.mean()}, Sum: {arr_from_range.sum()}, Standard deviation: {arr_from_range.std()}")

# NumPy Question 6
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.
arr_random_values = np.random.normal(loc=0, scale=1, size=200)
print(f"Array of 200 random values drawn from a normal distributio. Mean: {arr_random_values.mean()}, Standard deviation: {arr_random_values.std()}")


# --- Matplotlib ---

#Matplotlib Question 1
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Matplotlib Question 2
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title('Subject Scores')
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.show()

# Matplotlib Question 3
# Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color='yellow', label = 'Dataset 1')
plt.scatter(x2, y2, color='blue', label = 'Dataset 2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Dataset 1 vs Dataset 2')
plt.legend()
plt.show()

# Matplotlib Question 4
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side.
# In the left subplot, plot x vs y from Q1 as a line.
# In the right subplot, plot the subjects and scores from Q2 as a bar plot.
# Add a title to each subplot and call plt.tight_layout() before showing.
fig, axes = plt.subplots(1, 2)

axes[0].plot(x, y, color="blue")
axes[0].set_title("Squares")

axes[1].bar(subjects, scores, color="green")
axes[1].set_title("Subject Scores")

plt.tight_layout()
plt.show()


# --- Descriptive Stats ---

# Descriptive Stats Question 1
# Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
np.mean(data)
# Here I realized that it is more readable to print everything from a new line
print(f"Data list: {data}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")

# Descriptive Stats Question 2
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)).
# Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
arr_500_random_values = np.random.normal(65, 10, 500)
plt.hist(arr_500_random_values, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Question 3
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
import matplotlib.pyplot as plt

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], tick_labels=['Group A', 'Group B'])
plt.title("Score Comparison")
plt.show()

# Descriptive Stats Question 4
# You are given two datasets: one normally distributed and one 'exponential' distribution.
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
# Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.
plt.boxplot([normal_data, skewed_data], tick_labels=['Normal', 'Exponential'])
plt.title("Distribution Comparison")
plt.show()
# Exponential distribution is more skewed.
# For the normal distribution - the mean would be a more appropriate measure.
# For the exponential distribution - the median would be a more appropriate measure.

# Descriptive Stats Question 5
# Print the mean, median, and mode of the following:
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print(f"Data1: {data1}")
print(f"Mean: {np.mean(data1)}, Median: {np.median(data1)}, Mode: {stats.mode(data1).mode}")
print(f"Data2:  {data2}")
print(f"Mean: {np.mean(data2)}, Median: {np.median(data2)}, Mode: {stats.mode(data2).mode}")
# Why are the median and mean so different for data2? Add your answer as a comment in the code.
# Cause there is a huge abnormal value in the second dataset - 150 that impacts arithmetic average (mean) much more than the median.


# --- Hypothesis Testing ---

# Hypothesis Question 1
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_val  = stats.ttest_ind(group_a, group_b)
print(f"Independent samples t-test. T-statistic: {t_stat}, P-value: {p_val}")

# Hypothesis Question 2
# Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.
if p_val < 0.05:
    print("The difference is statistically significant at alpha = 0.05.")
else:
    print("No statistically significant difference detected at alpha = 0.05.")

# Hypothesis Question 3
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat, p_val  = stats.ttest_rel(before, after)
print(f"Paired t-test. T-statistic: {t_stat}, P-value: {p_val}")

# Hypothesis Question 4
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val  = stats.ttest_1samp(scores, 70)
print(f"One-sample t-test. T-statistic: {t_stat}, P-value: {p_val}")

# Hypothesis Question 5
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.
t_stat, p_val  = stats.ttest_ind(group_a, group_b, alternative='less')
print(f"One-tailed t-test (whether group_a scores are less than group_b scores). T-statistic: {t_stat}, P-value: {p_val}")

# Hypothesis Question 6
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("Conclusion: For Q1 t-test shows that there is a statistically significant difference between group_a and group_b scores.")
print("Group A scores are less than Group B scores. P-value is 1.5471178249432407e-06 that is less than chosen significance level (0.05).")
print("It means that it is almost unlikely that the difference is due to a random chance. Thus, Group B performs better than Group A.")


# --- Correlation ---

# Correlation Question 1
# Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).
# What do you expect the correlation to be, and why? Add your answer as a comment in the code.
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
# The correlation should be full positive as the difference between each on numbers in both list is a multiplication by 2
print(f"Correlation matrix: \n {corr_matrix}")
print(f"Correlation coefficient (position [0, 1]): {corr_matrix[0, 1]}")

# Correlation Question 2
# Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
corr_x_y = pearsonr(x, y)
print(f"Pearson correlation coefficient: {corr_x_y[0]}, P-value: {corr_x_y[1]}")

# Correlation Question 3
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
df_corr = df.corr()
print(f"DataFrame: \n {df}")
print(f"Correlation matrix for the DataFrame: \n {df_corr}")

# Correlation Question 4
# Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Correlation Question 5
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".
sns.heatmap(df_corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines ---

# Pipeline Question 1
# A data pipeline is a sequence of processing steps where each step takes in data, transforms it, and passes the result to the next.
# You don't need a special framework to build one -- chaining plain functions together is often enough.
#
# Given the array below, which contains some missing values scattered throughout:
#
# arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
#
# Implement the following three functions and then connect them in a data_pipeline() function.
#
# create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values".
# clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
# summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.
# data_pipeline(arr) -- calls the three functions above in sequence and returns the summary dictionary.
#
# Call data_pipeline(arr) and print each key and its value from the result.
# This is the last answer to put in warmups_01.py. Congrats!!!
# The next question will be in prefect_warmup.py, but will implement the same functionality using Prefect instead of plain Python.

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    cleaned = series.dropna()

    return cleaned

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    return summarize_data(clean_data(create_series(arr)))

result = data_pipeline(arr)
print("Summary of the data:")
for key, value in result.items():
    print(f"{key}: {value}")

