from dotenv import load_dotenv
from smolagents.tools import tool
import os
from openai import OpenAI
import pandas as pd
from scipy.stats import pearsonr

# Pre-task: Load the Data
#
# Your agent will need access to the World Happiness data. If you still have the merged file from Week 1, you can point directly to it:
#
DATA_PATH = '../assignments_01/outputs/merged_happiness.csv'
#
# If you don't have that file, you can load and merge the yearly CSVs from assignments/resources/happiness_project/ inside a load_happiness_data tool — see the hint in Task 1 below.

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

# --- Task 1: Define Your Tools ---

# Using the smolagents @tool decorator, implement the four tools below. Each tool operates on a shared global DataFrame (define df = None at the top of the file and update it inside load_happiness_data).

df = None

def _ensure_loaded() -> None:
    """Load the DataFrame from DATA_PATH if it has not been loaded yet."""
    global df
    if df is None:
        df = pd.read_csv(DATA_PATH, sep=',', decimal='.')

# Tool 1: load_happiness_data

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Reads the merged CSV from the default data path.
    Stores the result in a global df variable for use by other tools.

    Returns:
        dict: A dictionary with the following keys:
            - shape (tuple): The (rows, columns) dimensions of the loaded DataFrame.
            - columns (list[str]): The list of column names in the DataFrame.

    Example:
        >>> load_happiness_data()
        {"shape": (1704, 10), "columns": ["country", "year", "Happiness score", ...]}
    """
    # Load the merged CSV from DATA_PATH. If that file does not exist, fall back to loading and merging all yearly CSVs from assignments/resources/happiness_project/ (use a loop, just like in the Week 1 project).
    # Store the result in the global df. Return a dict with "shape" and "columns".
    global df

    if df is not None:
        print(f"Data have been already loaded. Returning results. Shape: {df.shape}")

        return {
            "shape": df.shape,
            "columns": df.columns.tolist()
        }

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=',', decimal='.')
    print(f"Data loaded successfully. Shape: {df.shape}")
    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }

# Tool 2: summarize_column

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Computes count, mean, standard deviation, min, max, and quartile values
    for the specified numeric column using pandas describe().

    Args:
        column (str): The name of the column to summarize. Must exist in the
            loaded DataFrame.

    Returns:
        dict: Descriptive statistics for the column (count, mean, std, min,
            25%, 50%, 75%, max). Returns {"error": "..."} if no data has been
            loaded or the column name is not found.

    Example:
        >>> summarize_column("Happiness score")
        {"count": 1704.0, "mean": 5.47, "std": 1.11, "min": 2.37, ...}
    """
    if df is None or column not in df.columns:
        _ensure_loaded()
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {df.columns.tolist()}"}

    return df[column].describe().to_dict()
    # Return df[column].describe().to_dict(). Return {"error": "..."} if no data is loaded or the column is not found.

# Tool 3: compute_correlation

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Uses scipy.stats.pearsonr to measure the linear relationship between two
    columns in the loaded DataFrame. Both columns must be numeric and present
    in the dataset.

    Args:
        col1 (str): The name of the first numeric column.
        col2 (str): The name of the second numeric column.

    Returns:
        dict: A dictionary with the following keys:
            - col1 (str): Name of the first column.
            - col2 (str): Name of the second column.
            - pearson_r (float): Pearson correlation coefficient, rounded to 4 decimal places.
            - p_value (float): Two-tailed p-value for the correlation, rounded to 4 decimal places.
            Returns {"error": "..."} if no data is loaded, columns are not found, or
            computation fails.

    Example:
        >>> compute_correlation("GDP per capita", "Happiness score")
        {"col1": "GDP per capita", "col2": "'"Happiness score""pearson_r": 0.7892, "p_value": 0.0}
    """
    _ensure_loaded()
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"Columns not found. Available columns: {df.columns.tolist()}"}

    try:
        pearson_r, p_value = pearsonr(df[col1], df[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(pearson_r, 4),
            "p_value": round(p_value, 4),
        }
    except Exception as e:
        return {"error": str(e)}

    # Use scipy.stats.pearsonr. Return a dict with "col1", "col2", "pearson_r", and "p_value" (rounded to 4 decimal places). Return {"error": "..."} on bad input.

# Tool 4: get_top_n_countries

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the loaded DataFrame to the specified year, sorts by the given column
    in descending order, and returns the top N rows as a list of records.

    Args:
        column (str): The name of the column to rank countries by (e.g., "Happiness score").
        year (int): The year to filter the dataset on (e.g., 2020).
        n (int, optional): The number of top countries to return. Defaults to 5.

    Returns:
        dict: A list of dicts, each containing "country" and the value of the
            requested column. Returns {"error": "..."} if no data is loaded,
            the column is not found, or no data exists for the given year.

    Example:
        >>> get_top_n_countries("Happiness score", 2020, n=3)
        [{"Country": "Finland", "Happiness score": 7.809}, ...]
    """
    _ensure_loaded()
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {df.columns.tolist()}"}

    df_year = df[df["Year"] == year]
    if df_year.empty:
        return {"error": "No data for the specified year."}

    top_n = df_year.nlargest(n, column)

    return top_n[["Country", column]].to_dict(orient="records")

# Tool 5: get_dataframe_records

# I created this function to help with very annoying error when superagent could't get access to global df cause of a sandbox env it was running in
@tool
def get_dataframe_records(columns: list | None = None, year: int | None = None) -> list:
    """Return rows from the loaded DataFrame as a list of dictionaries.

    Use this tool when you need raw data to build a custom plot or perform a
    computation that the other tools don't cover. The DataFrame itself is NOT
    accessible from your code sandbox — call this tool to get the rows instead.

    Args:
        columns (list[str] | None): Optional list of columns to include. If None,
            all columns are returned.
        year (int | None): Optional year to filter on (matches the "Year" column).
            If None, all years are returned.

    Returns:
        list[dict]: One dict per row, keyed by column name. Returns
            [{"error": "..."}] if a requested column is missing.

    Example:
        >>> get_dataframe_records(columns=["Country", "Year", "Happiness score", "Regional indicator"], year=2023)
        [{"Country": "Finland", "Year": 2023, "Happiness score": 7.804, "Regional indicator": "Western Europe"}, ...]
    """
    _ensure_loaded()
    data = df
    if year is not None:
        data = data[data["Year"] == year]
    if columns is not None:
        missing = [c for c in columns if c not in data.columns]
        if missing:
            return [{"error": f"Columns not found: {missing}. Available: {df.columns.tolist()}"}]
        data = data[columns]
    return data.to_dict(orient="records")

# Filter df to the given year, sort by column in descending order, and return the top n rows as a list of dicts (each dict has "country" and the requested column value). Return {"error": "..."} on bad input.
#
# Write complete Google-style docstrings for all four tools. Remember: smolagents reads your docstring to understand what the tool does and when to use it.



# --- Task 2: Build the Agent ---

# Instantiate a CodeAgent:

from smolagents import CodeAgent, OpenAIServerModel, tool

model = OpenAIServerModel(api_key=os.getenv("OPENAI_API_KEY"), model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

IMPORTANT: The module-level `df` DataFrame is NOT visible from your code sandbox.
Never reference a bare `df` variable in your code — it will raise NameError.
To get raw rows for custom plots or computations, call `get_dataframe_records(...)`
and build a local DataFrame in your sandbox, e.g.:

    rows = get_dataframe_records(columns=["Country", "Year", "Happiness score", "Regional indicator"])
    import pandas as pd
    local_df = pd.DataFrame(rows)

The dataset's region column is named "Regional indicator" (not "Region").
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries, get_dataframe_records],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats", "numpy"],
    max_steps=8,
)

# --- Task 5: Reflection ---
# Add a comment block at the very bottom of project_07.py answering these three questions:
#
# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
#    The agent called compute_correlation("GDP per capita", "Happiness score"), received a pearson_r ≈ 0.6218 and a p_value ≈ 0.0,
#    and just said that the correlation is "statistically significant"
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
#    Query 5 (the regional line chart) was more capable than expected.
#    Without any hint about which column encodes region, the agent correctly inferred it was "Regional indicator",
#    grouped the DataFrame by both year and region, computed the mean happiness score per group,
#    and produced clean matplotlib code that looped over regions — all in one step, with no errors and a properly saved PNG.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
#    A filter_data(column, operator, value) tool would be very useful.
#    It would return a filtered slice of the DataFrame (e.g., rows where "Year" == 2022 or "Happiness score" > 6.0) without the agent having to write raw pandas code.
#    This would let the agent answer questions like "Which countries had a happiness score above 7 every year from 2015 to 2023?" or "Show me only Sub-Saharan African countries in 2021" using a safe, declarative tool call instead of generating code.


# --- Running the Project ---
#
# Structure your file so all setup and queries run when the script is executed directly:

if __name__ == "__main__":
    # --- Task 3: Run Guided Queries ---
    # Run the five queries below in sequence. Use reset=False so the agent retains context across turns. Print each response.

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the 'Happiness score' column.",
        "What is the correlation between 'GDP per capita' and 'Happiness score'? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot 'Happiness score' over the years as a line chart, with one line per region. Save the plot to ../assignments_07/outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # Query 5 should cause the agent to write matplotlib code (no tool covers multi-line regional plots).
    # Verify that outputs/happiness_by_region.png is saved to disk after running.


    # --- Task 4: Your Own Questions ---
    #
    # Run two additional queries of your own choice. Try to make at least one of them require the agent to write code rather than just call a tool.

    # My query 1
    my_query_1 = "What are the top 5 countries by GDP per capita in 2019, and how do their happiness scores compare?"
    print(f"\n--- Query: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered tool use — get_top_n_countries was called twice (once for GDP per capita, once for Happiness scores).
    # Then a bunch of code generation, mainly for data manipulations.
    # Then get_top_n_countries again for the list of all countries to comparison, and again some scripting for datamanipulation.

    # My query 2
    my_query_2 = "Create a scatter plot of 'GDP per capita' vs 'Happiness score' for the year 2023, color each point by region, add a trend line, label the axes clearly, and save the plot to ../assignments_07/outputs/gdp_vs_happiness_2023.png."
    print(f"\n--- Query: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered both tool use (load data check, get_top_n_countries) and code generation.
    # The agent wrote matplotlib/seaborn code (with a bunch of data manipulations) to build the scatter plot with color-coded regions and a trend line.
    # This is cause no existing tool covers custom multi-variable visualizations.

# The full project should be runnable with:
#
# python project_07.py
#
# When you run it, all five guided queries should complete, the plot should be saved to outputs/happiness_by_region.png, and your two custom queries should run as well.
