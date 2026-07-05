from pathlib import Path
import pandas as pd
import re
from pandas import DataFrame
from prefect import flow, task
from prefect.logging import get_run_logger
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
import pprint

# Task 1: Load Multiple Years of Data
#
# Load data from all ten yearly CSV files into a single DataFrame. Your implementation should not duplicate code for each year -- iterate over a list of file paths and load them in a loop.

@task(retries=3, retry_delay_seconds=2)
def load_csv_data_to_df(path_to_files):
    logger = get_run_logger()

    logger.info('Reading target folder')
    world_happiness_files = list(Path(path_to_files).glob('*.csv'))
    logger.info(f'Found: \n {(file.name for file in world_happiness_files)}')

    dfs = list()
    logger.info('Loading files to df')
    for file in world_happiness_files:
        file_name = file.name
        df = pd.read_csv(f'{path_to_files}/{file_name}', sep=';', decimal=',')
        df['Year'] = extract_year_from_string(file_name)
        dfs.append(df)
        logger.info(f'{file_name} was loaded')

    logger.info('All files were loaded')
    print(f"fdfs: {dfs}")

    return pd.concat(dfs)

def extract_year_from_string(file_name):
    year = None
    match = re.search(r'\d{4}', file_name)
    if match:
        year = match.group(0)

    return year

@task(retries=3, retry_delay_seconds=2)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    '''
    Basic cleaning:
      1) Ensure Happiness score is numeric.
      2) Drop missing values.
    '''
    df = df.copy()
    df['Happiness score'] = pd.to_numeric(df['Happiness score'], errors='coerce')
    df['GDP per capita'] = pd.to_numeric(df['GDP per capita'], errors='coerce')

    df = df.dropna(subset=['Happiness score', 'Year', 'GDP per capita'])
    logger.info('Data were cleaned')

    return df

# You discovered some quirks when you inspected the raw files. Make sure you account for those when calling pd.read_csv(). There is also something missing from each file that you will need to add before merging: each row needs to know which year it came from. Think about where to add that information.
#
# After loading and merging, save the combined dataset to:
#
# assignments_01/outputs/merged_happiness.csv
#
# Add retries=3, retry_delay_seconds=2 to this task's decorator. File I/O is exactly the kind of operation that can fail intermittently in production pipelines, and this is where retries earn their keep.@tasksk(retries=3, retry_delay_seconds=2)
@task(retries=3, retry_delay_seconds=3)
def save_df_to_csv(df: DataFrame):
    logger = get_run_logger()

    csv_file_name = 'outputs/merged_happiness.csv'
    df.to_csv(csv_file_name, index=False)
    logger.info(f'All files were saved to {csv_file_name}')


# Task 2: Descriptive Statistics
# Compute and log overall descriptive statistics for happiness_score: mean, median, and standard deviation.
# Then compute and log the mean happiness score grouped by year and by region. Looking at the regional breakdown is often the most interesting part of this dataset -- you may already have a hypothesis about which regions rank highest before you run the numbers.

@task
def compute_descriptive_statistics(df):
    logger = get_run_logger()

    logger.info('Compute descriptive statistics')
    mean = df['Happiness score'].mean()
    median = df['Happiness score'].median()
    std = df['Happiness score'].std()
    logger.info(f'Overall happiness score mean: {mean}')
    logger.info(f'Overall happiness score median: {median}')
    logger.info(f'Overall happiness score std: {std}')

    mean_happiness_by_year = df.groupby('Year')['Happiness score'].mean()
    mean_happiness_by_region = df.groupby('Regional indicator')['Happiness score'].mean()
    logger.info(f'Mean happiness by year: \n {pprint.pformat(mean_happiness_by_year.to_dict())}')
    logger.info(f'Mean happiness by region: \n {pprint.pformat(mean_happiness_by_region.to_dict())}')

    return {
        'overall_stats': {
            'mean': float(mean),
            'median': float(median),
            'std': float(std),
        },
        'by_year': mean_happiness_by_year.to_dict(),
        'by_region': mean_happiness_by_region.to_dict(),
    }

# Task 3: Visual Exploration
# Create and save the following visualizations to assignments_01/outputs/:
#
# A histogram of all happiness scores across all years. Save as happiness_histogram.png.
# A boxplot comparing happiness score distributions across years (one box per year). Save as happiness_by_year.png.
# A scatter plot showing the relationship between GDP per capita and happiness score. Save as gdp_vs_happiness.png.
# A correlation heatmap (using sns.heatmap() with annot=True) showing the Pearson correlations between all numeric columns. Save as correlation_heatmap.png.
#
# Log a message after each plot is saved so you can see the progress in the Prefect dashboard.

@task
def describe_and_plot(df: pd.DataFrame):
    logger = get_run_logger()

    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    plot_df = df.copy()

    plt.hist(plot_df['Happiness score'], bins=30)
    plt.title('Distribution of Happiness score across all years')
    plt.xlabel('Happiness score')
    plt.ylabel('Count')
    plt.tight_layout()
    histogram_file = f'{output_dir}/happiness_histogram.png'
    plt.savefig(histogram_file, dpi=150)
    plt.close()
    logger.info(f'Histogram saved to {histogram_file}')

    year_order = sorted(plot_df['Year'].unique(), key=int)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df[['Year', 'Happiness score']], x='Year', y='Happiness score', order=year_order)
    plt.title('Happiness score by year')
    plt.xlabel('Year')
    plt.ylabel('Happiness score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    by_year_file = f'{output_dir}/happiness_by_year.png'
    plt.savefig(by_year_file, dpi=150)
    plt.close()
    logger.info(f'Boxplot saved to {by_year_file}')

    plt.scatter(plot_df['GDP per capita'], plot_df['Happiness score'])
    plt.title('GDP per capita vs Happiness score')
    plt.xlabel('GDP per capita')
    plt.ylabel('Happiness score')
    plt.tight_layout()
    scatter_file = f'{output_dir}/gdp_vs_happiness.png'
    plt.savefig(scatter_file, dpi=150)
    plt.close()
    logger.info(f'Scatter plot saved to {scatter_file}')

    # Keep only columns that have at least some numeric data
    numeric_df = df.select_dtypes(include='number')
    df_corr = numeric_df.corr(method='pearson')
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_corr, annot=True)
    plt.title('Correlation heatmap (Numeric Columns)')
    plt.tight_layout()
    heatmap_file = f'{output_dir}/correlation_heatmap.png'
    plt.savefig(heatmap_file, dpi=150)
    plt.close()
    logger.info(f'Correlation heatmap saved to {heatmap_file}')

# Task 4: Hypothesis Testing
#
# The pandemic began in early 2020. Did it affect global happiness scores? Test this directly: run an independent samples t-test comparing happiness scores from 2019 to 2020.
#
# Log the t-statistic, p-value, the mean happiness for each group, and a plain-language interpretation of the result at alpha = 0.05. Your interpretation should say something meaningful -- not just "we reject the null hypothesis" but what that actually means in terms of this data.

@task
def run_ttest(df):
    logger = get_run_logger()

    logger.info(f'Running an independent samples t-test comparing happiness scores from 2019 to 2020.')
    happiness_scores_2019 = df.loc[df['Year'] == '2019', 'Happiness score']
    happiness_scores_2020 = df.loc[df['Year'] == '2020', 'Happiness score']
    t_stat, p_val = stats.ttest_ind(happiness_scores_2019, happiness_scores_2020)
    logger.info(f'Independent samples t-test. T-statistic: {t_stat}, P-value: {p_val}')

    if p_val < 0.05:
        logger.info(f'The difference is statistically significant at alpha = 0.05.')
    else:
        logger.info(f'No statistically significant difference detected at alpha = 0.05.')
        logger.info(f'Conclusion: An independent t-test comparing 2019 and 2020 happiness score shows that there is no statistically significant difference (t_stat = {t_stat}, p_val = {p_val}, alpha = 0.05).')
        logger.info('The negative t_stat suggests 2019 year mean is a bit lower than 2020 one, but again that difference is small and likely is due to random variation.')

# Add a second test of your choice (for example, comparing two specific regions that you expect to differ based on the descriptive statistics you computed earlier).

    logger.info('Comparing two specific regions that I expect to differ based on the descriptive statistics I computed earlier.')

    top_region = 'North America and ANZ'
    bottom_region = 'Commonwealth of Independent States'
    logger.info(f'Selecting "{top_region}" and "{bottom_region}" as the most different regions based on "Mean happiness by region" results (7.1761944444444445 and 5.35164947368421 correspondently).')
    top_region_scores = df.loc[df['Regional indicator'] == top_region, 'Happiness score']
    bottom_region_scores = df.loc[df['Regional indicator'] == bottom_region, 'Happiness score']

    region_t_stat, region_p_val = stats.ttest_ind(top_region_scores, bottom_region_scores)
    logger.info(f'Mean happiness by group ({top_region} vs {bottom_region}): {top_region_scores.mean()} vs {bottom_region_scores.mean()}')
    logger.info(f'Second independent samples t-test. T-statistic: {region_t_stat}, P-value: {region_p_val}')

    if region_p_val < 0.05:
        logger.info(f'The difference between {top_region} and {bottom_region} is statistically significant at alpha = 0.05.')
        logger.info(f'Conclusion: the extremely small p-value and huge positive t-statistic confirm that {top_region} (mean = {top_region_scores.mean()}) is consistently and much more happier than the {bottom_region} (mean = {bottom_region_scores.mean()}) across all years in the dataset.')
    else:
        logger.info(f'No statistically significant difference detected at alpha = 0.05.')

    return {
        'years_2019_2020': {
            't_stat': float(t_stat),
            'p_val': float(p_val),
            'mean_2019': float(happiness_scores_2019.mean()),
            'mean_2020': float(happiness_scores_2020.mean()),
        },
        'regions': {
            'region_a': top_region,
            'region_b': bottom_region,
            't_stat': float(region_t_stat),
            'p_val': float(region_p_val),
            'mean_region_a': float(top_region_scores.mean()),
            'mean_region_b': float(bottom_region_scores.mean()),
        },
    }

# Task 5: Correlation and Multiple Comparisons
# For each numeric explanatory variable, compute the Pearson correlation with happiness score using scipy.stats.pearsonr and log the coefficient and p-value.
#
# Each time you run a statistical test at alpha = 0.05, you accept a 5% chance of a false positive -- concluding a relationship is real when it isn't. That's a reasonable risk for a single test. But when you run many tests at once, those small risks add up. If you run 20 independent tests, you'd expect roughly one false positive just by chance, even if none of the relationships are actually real. The more tests you run, the more likely you are to stumble onto something that looks significant but isn't.
# This is called the multiple comparisons problem, and it's one of the most common sources of misleading findings in data analysis. A simple and widely used fix is the Bonferroni correction: divide your significance threshold by the number of tests you ran.
#
# Count how many correlation tests you performed, then compute:
#
# adjusted_alpha = 0.05 / number_of_tests
#
# Log which correlations are significant at the original alpha = 0.05, and which remain significant after applying the correction. You may find that some results that looked significant at first don't hold up under the stricter threshold -- that's a useful finding in itself.

@task
def compute_correlations(df: pd.DataFrame) -> dict:
    logger = get_run_logger()

    numeric_df = df.select_dtypes(include='number')
    explanatory_cols = [col for col in numeric_df.columns if col != 'Happiness score']

    results = {}
    skipped = {}
    logger.info(f'Computing Pearson correlations for {len(explanatory_cols)} variables against Happiness score.')

    for col in explanatory_cols:
        clean = df[['Happiness score', col]].dropna()

        # Here I needed to do some dirty work to make sure the app is not crushing cause
        # pearsonr requires at least 2 paired points and non-constant input
        if len(clean) < 2:
            skipped[col] = 'fewer than 2 paired non-null values'
            logger.info(f'Skipping {col}: {skipped[col]}.')
            continue

        if clean['Happiness score'].nunique() < 2 or clean[col].nunique() < 2:
            skipped[col] = 'one of the variables is constant after dropna'
            logger.info(f'Skipping {col}: {skipped[col]}.')
            continue

        pearson_corr_coef, p_value = pearsonr(clean['Happiness score'], clean[col])
        results[col] = {'r': float(pearson_corr_coef), 'p_val': float(p_value)}
        logger.info(f'{col}: r = {pearson_corr_coef}, p_val = {p_value}')

    n_tests = len(results)
    if n_tests == 0:
        logger.info('No valid correlation tests were run after filtering; skipping Bonferroni correction.')
        return {
            'correlations': {},
            'adjusted_alpha': None,
            'bonferroni_significant': {},
            'strongest': None,
            'skipped': skipped,
        }

    adjusted_alpha = 0.05 / n_tests
    logger.info(f'Number of tests: {n_tests}. Bonferroni adjusted alpha: {adjusted_alpha}')

    logger.info('Significant at original alpha = 0.05')
    for col, vals in results.items():
        if vals['p_val'] < 0.05:
            logger.info(f'{col}: r = {vals["r"]}, p = {vals["p_val"]}  ✓')

    logger.info(f'Still significant after Bonferroni correction (alpha = {adjusted_alpha})')
    bonferroni_significant = {}
    for col, vals in results.items():
        if vals['p_val'] < adjusted_alpha:
            bonferroni_significant[col] = vals
            logger.info(f'{col}: r = {vals["r"]}, p = {vals["p_val"]}  ✓')

    logger.info('Dropped after Bonferroni correction')
    for col, vals in results.items():
        if 0.05 > vals['p_val'] >= adjusted_alpha:
            logger.info(f'  {col}: r = {vals["r"]}, p = {vals["p_val"]}  ✗ (no longer significant)')

    strongest = max(results, key=lambda col: abs(results[col]['r']))
    logger.info(f'Strongest correlation with Happiness score (after Bonferroni): "{strongest}" (r = {results[strongest]["r"]:.4f})')

    return {
        'correlations': results,
        'adjusted_alpha': adjusted_alpha,
        'bonferroni_significant': bonferroni_significant,
        'strongest': strongest,
        'skipped': skipped,
    }


# Task 6: Summary Report
#
# Your final task should log a human-readable summary of the key findings from the entire pipeline. Think of it as the "report" step from the lesson -- the thing you'd share with a non-technical colleague. It should include:
#
# Total number of countries and years in the merged dataset.
# The top 3 and bottom 3 regions by mean happiness score.
# The result of the pre/post-2020 t-test in plain language.
# The variable most strongly correlated with happiness score (after Bonferroni correction).
#
# Log each of these as a separate logger.info() message so they're easy to find in the Prefect dashboard.

@task
def report_results(result: dict, clean_df: pd.DataFrame, corr_result: dict):
    logger = get_run_logger()

    logger.info('----------------------------------------')
    logger.info('Summary report: World Happiness Analysis')
    logger.info('----------------------------------------')

    unique_countries = clean_df['Country'].nunique()
    unique_years = clean_df['Year'].nunique()
    logger.info(f'Total unique countries: {unique_countries}')
    logger.info(f'Total unique years: {unique_years}')
    logger.info(f'Total observations: {len(clean_df)}')

    logger.info('Regional Breakdown')
    region_means = clean_df.groupby('Regional indicator')['Happiness score'].mean().sort_values(ascending=False)

    logger.info('Top 3 regions by mean happiness score:')
    for i, (region, score) in enumerate(region_means.head(3).items(), 1):
        logger.info(f'{i}. {region}: {score:.4f}')

    logger.info('Bottom 3 regions by mean happiness score:')
    for i, (region, score) in enumerate(region_means.tail(3).iloc[::-1].items(), 1):
        logger.info(f'{i}. {region}: {score:.4f}')

    logger.info('2019 vs 2020 Pandemic Impact Test')
    years_result = result['years_2019_2020']
    mean_2019 = years_result['mean_2019']
    mean_2020 = years_result['mean_2020']
    t_stat = years_result['t_stat']
    p_val = years_result['p_val']

    logger.info(f'Mean happiness 2019: {mean_2019:.4f}')
    logger.info(f'Mean happiness 2020: {mean_2020:.4f}')
    logger.info(f'T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}')
    logger.info(f'Conclusion: An independent t-test found no statistically significant difference in global happiness between 2019 and 2020 (p_val = {p_val}, alpha = 0.05).')
    logger.info(f'While the mean changed from {mean_2019:} to {mean_2020:}, this change could easily be due to random variation rather than a real pandemic effect.')

    logger.info('Strongest Correlation (after Bonferroni Correction)')
    if corr_result['strongest'] is not None:
        strongest_var = corr_result['strongest']
        strongest_r = corr_result['correlations'][strongest_var]['r']
        strongest_p = corr_result['correlations'][strongest_var]['p_val']
        adjusted_alpha = corr_result['adjusted_alpha']

        logger.info(f'Variable: {strongest_var}')
        logger.info(f'Pearson r: {strongest_r:.4f}')
        logger.info(f'P-value: {strongest_p:.4e}')
        logger.info(f'Bonferroni adjusted alpha: {adjusted_alpha:.6f}')
        logger.info(f'This variable shows the strongest association with happiness score and remains significant even after the Bonferroni correction for multiple comparisons.')
    else:
        logger.info('No variables remained significantly correlated after Bonferroni correction.')

    logger.info('----------------------------------------')
    logger.info('End of report: World Happiness Analysis')
    logger.info('----------------------------------------')


@flow
def analysis_pipeline(path_to_data):
    df = load_csv_data_to_df(path_to_data)
    clean_df = clean_data(df)
    save_df_to_csv(clean_df)
    compute_descriptive_statistics(clean_df)
    describe_and_plot(clean_df)
    ttest_result = run_ttest(clean_df)
    corr_result = compute_correlations(clean_df)
    report_results(ttest_result, clean_df, corr_result)

if __name__ == "__main__":
    analysis_pipeline('happiness_project')
