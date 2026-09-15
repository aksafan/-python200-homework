from dotenv import load_dotenv
from smolagents.tools import tool
import os
import re
from pathlib import Path
from openai import OpenAI
import pandas as pd
from scipy.stats import pearsonr

# Pre-task: Load the Data
#
# Your agent will need access to the World Happiness data. If you still have the merged file from Week 1, you can point directly to it:
#
# I added all folder vars below to make the script runs from any working directory and AI reviewer is less likely to hallucinate like it did a couple of times already
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'assignments_01' / 'outputs' / 'merged_happiness.csv'
# Fallback in case the merged file is missing
FALLBACK_DIR = BASE_DIR.parent / 'assignments_01' / 'happiness_project'
# Plots go here regardless of where the script was launched from
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
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

def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Snake_case the column names so tools, queries and output keys share one convention."""
    frame.columns = [c.strip().lower().replace(' ', '_') for c in frame.columns]
    return frame

def _read_data() -> pd.DataFrame:
    """Read the merged CSV, or merge the yearly CSVs when it is not there."""
    if DATA_PATH.exists():
        print(f"Loading data from {DATA_PATH}...")
        return _normalize_columns(pd.read_csv(DATA_PATH, sep=',', decimal='.'))

    # Same loop as the Week 1 pipeline: one file per year, semicolon separated with
    # comma decimals, and the year taken from the filename.
    print(f"{DATA_PATH} not found. Merging yearly CSVs from {FALLBACK_DIR}...")
    frames = []
    for path in sorted(FALLBACK_DIR.glob('*.csv')):
        match = re.search(r'\d{4}', path.name)
        if match is None:
            continue
        part = pd.read_csv(path, sep=';', decimal=',')
        part['Year'] = int(match.group(0))
        frames.append(part)

    if not frames:
        raise FileNotFoundError(f"No yearly CSVs found in {FALLBACK_DIR}")

    return _normalize_columns(pd.concat(frames, ignore_index=True))

def _ensure_loaded() -> None:
    """Load the DataFrame if it has not been loaded yet."""
    global df
    if df is None:
        df = _read_data()

# Tool 1: load_happiness_data

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Reads the merged CSV from DATA_PATH. If that file does not exist, falls back to
    reading and concatenating every yearly CSV in FALLBACK_DIR, just like the Week 1
    project. Column names are normalised to snake_case either way.
    Stores the result in a global df variable for use by other tools.

    Returns:
        dict: A dictionary with the following keys:
            - shape (tuple): The (rows, columns) dimensions of the loaded DataFrame.
            - columns (list[str]): The list of column names in the DataFrame.

    Example:
        >>> load_happiness_data()
        {"shape": (1362, 12), "columns": ["ranking", "country", "happiness_score", ...]}
    """
    global df

    if df is not None:
        print(f"Data have been already loaded. Returning results. Shape: {df.shape}")

        return {
            "shape": df.shape,
            "columns": df.columns.tolist()
        }

    df = _read_data()
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
        >>> summarize_column("happiness_score")
        {"count": 1362.0, "mean": 5.44, "std": 1.12, "min": 1.859, ...}
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
        >>> compute_correlation("gdp_per_capita", "happiness_score")
        {"col1": "gdp_per_capita", "col2": "happiness_score", "pearson_r": 0.6218, "p_value": 0.0}
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
        column (str): The name of the column to rank countries by (e.g., "happiness_score").
        year (int): The year to filter the dataset on (e.g., 2020).
        n (int, optional): The number of top countries to return. Defaults to 5.

    Returns:
        dict: A list of dicts, each containing "country" and the value of the
            requested column. Returns {"error": "..."} if no data is loaded,
            the column is not found, or no data exists for the given year.

    Example:
        >>> get_top_n_countries("happiness_score", 2020, n=3)
        [{"country": "Finland", "happiness_score": 7.809}, ...]
    """
    _ensure_loaded()
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {df.columns.tolist()}"}

    df_year = df[df["year"] == year]
    if df_year.empty:
        return {"error": "No data for the specified year."}

    top_n = df_year.nlargest(n, column)

    return top_n[["country", column]].to_dict(orient="records")

# Tool 5: get_dataframe_records

# I created this function to help with very annoying error when superagent could't get access to global df cause of a sandbox env it was running in
@tool
def get_dataframe_records(columns: list | None = None, year: int | None = None) -> list:
    """Return rows from the loaded DataFrame as a list of dictionaries.

    Use this tool when you need raw data to build a custom plot or perform a
    computation that the other tools don't cover. The DataFrame itself is NOT
    accessible from your code sandbox — call this tool to get the rows instead.

    Args:
        columns (list[str] | None): Optional list of columns to include. If None
            or empty, all columns are returned.
        year (int | None): Optional year to filter on (matches the "year" column).
            If None, all years are returned.

    Returns:
        list[dict]: One dict per row, keyed by column name. Returns
            [{"error": "..."}] if a requested column is missing.

    Example:
        >>> get_dataframe_records(columns=["country", "year", "happiness_score", "regional_indicator"], year=2023)
        [{"country": "Finland", "year": 2023, "happiness_score": 7.804, "regional_indicator": "Western Europe"}, ...]
    """
    _ensure_loaded()
    data = df
    if year is not None:
        data = data[data["year"] == year]
    # `if columns:` not `if columns is not None:` - an agent that means "give me
    # everything" often passes columns=[], and df[[]] silently returns zero columns.
    if columns:
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

    rows = get_dataframe_records(columns=["country", "year", "happiness_score", "regional_indicator"])
    import pandas as pd
    local_df = pd.DataFrame(rows)

Column names are snake_case. The region column is named "regional_indicator" (not "region").

When a query asks you to save a plot to outputs/<name>.png, save it to this absolute
directory instead, so the file lands in the right place no matter where the script was
launched from:
"""

SYSTEM_PROMPT += f"    {OUTPUT_DIR}\n"

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
#    The agent called compute_correlation("gdp_per_capita", "happiness_score") and returned
#    {'pearson_r': 0.6218, 'p_value': 0.0, 'statistical_significance': 'significant'} - it added that
#    third key itself, the tool never returns it.
#    It used the p-value correctly in direction (a p-value that small does rule out "no linear
#    relationship"), but it never stated the threshold out loud - it applied the conventional
#    alpha = 0.05 implicitly. Worth noting it did not qualify that with n = 1362 rows almost any
#    non-zero r clears that threshold, so here the p-value is really reporting sample size; the
#    r ≈ 0.62 is the number that says how strong the relationship is, and the agent did not
#    draw that distinction.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
#    Less capable, and in a way I did not anticipate. On an earlier run, get_dataframe_records had a
#    bug: an agent that wants every column tends to pass columns=[], and df[[]] silently returns zero
#    columns, so the tool answered "0 records". The agent concluded "The World Happiness dataset is
#    currently empty" - a reasonable read of what it observed. The part that surprised me is what came
#    next: because these queries run with reset=False, that conclusion stayed in memory, and the agent
#    opened all six remaining queries by restating it and calling final_answer immediately. It never
#    re-checked, never tried a different tool, never doubted its own earlier observation. One bad tool
#    return poisoned the whole session. Agents accumulate beliefs, and a wrong one is sticky.
#
#    The flip side, on the fixed run: in my second custom query the agent's first attempt crashed on
#    np.polyfit because it had used numpy without importing it. It read the traceback, re-emitted the
#    whole code block with "import numpy as np" added, and the trend line came out fine. Recovering
#    from its own traceback without being told to was more capable than I expected.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
#    A filter_data(column, operator, value) tool would be very useful.
#    It would return a filtered slice of the DataFrame (e.g., rows where year == 2022 or happiness_score > 6.0) without the agent having to write raw pandas code.
#    This would let the agent answer questions like "Which countries had a happiness score above 7 every year from 2015 to 2023?" or "Show me only Sub-Saharan African countries in 2021" using a safe, declarative tool call instead of generating code.


# --- Running the Project ---
#
# Structure your file so all setup and queries run when the script is executed directly:

if __name__ == "__main__":
    # --- Task 3: Run Guided Queries ---
    # Run the five queries below in sequence. Use reset=False so the agent retains context across turns. Print each response.

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
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
    my_query_1 = "What are the top 5 countries by gdp_per_capita in 2019, and how do their happiness_score values compare?"
    print(f"\n--- Query: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # Both. Step 1 was a tool call: get_top_n_countries(column="gdp_per_capita", year=2019, n=5).
    # The ranking half is exactly what a tool is for. The "how do their happiness scores compare" half
    # has no tool, so step 2 pulled the raw rows with get_dataframe_records(columns=["country",
    # "happiness_score"], year=2019) and step 3 was generated code that matched the two lists up.
    # Notably Qatar tops GDP per capita (10.0) but scores 6.37, below Luxembourg's 7.09 - the agent
    # had to write code to surface that, no tool would have.

    # My query 2
    my_query_2 = "Create a scatter plot of gdp_per_capita vs happiness_score for the year 2023, color each point by region, add a trend line, label the axes clearly, and save the plot to outputs/gdp_vs_happiness_2023.png."
    print(f"\n--- Query: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # Both, and it took three steps. get_dataframe_records supplied the 2023 rows, then the agent wrote
    # matplotlib code that looped over regions for the per-region colors and used np.polyfit for the
    # trend line. That first attempt failed - it called np.polyfit without importing numpy - so it read
    # the error, re-emitted the block with the import added, and saved the PNG. No tool covers a
    # multi-series scatter with a fitted line, so this is the case where a CodeAgent genuinely earns
    # its keep over a ToolCallingAgent.

# The full project should be runnable with:
#
# python project_07.py
#
# When you run it, all five guided queries should complete, the plot should be saved to outputs/happiness_by_region.png, and your two custom queries should run as well.
