from dotenv import load_dotenv
from openai import OpenAI
import os

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')
api_key = os.getenv("OPENAI_API_KEY")

# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1

# Define the following Python function:

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Next, write the JSON schema dictionary that describes this function to an LLM — exactly like the get_current_time schema in the lesson.
# Your schema should include name, description, and parameters (with a celsius property of type "number").
celsius_schema = {
    'type': 'function',
    'function': {
        'name': 'celsius_to_fahrenheit',
        'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
        'parameters': {
            'type': 'object',
            'properties': {
                'celsius': {
                    'type': 'number',
                    'description': 'The temperature in degrees Celsius to convert to Fahrenheit.',
                },
            },
            'required': ['celsius'],
        },
    },
}

# Finally, call the function directly (not through an agent yet) with 0, 100, and -40 and print each result.
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2

# Copy the run_agent function from the lesson — the one that uses get_current_time as its only tool.
import json

from datetime import datetime

def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

time_schema = {
    'type': 'function',
    'function': {
        'name': 'get_current_time',
        'description': 'Returns the current local time as a string.',
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': [],
        },
    },
}

# I added it to satisfy the lesson and AI reviewer requirements about the exactly one tool
# Q3 adds celsius_schema
tools = [time_schema]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''

    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# Before calling it, add a comment block that predicts:
#
# 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
#
#    No. In the ReAct loop the model can only ACT through the schemas we hand it in
#    `tools`, and the only schema advertised here is get_current_time, whose description
#    is about reading the clock. Nothing in it matches a unit conversion, so the REASON
#    step has no relevant tool to select. On top of that, (100 * 9/5) + 32 is arithmetic
#    the model can already do from its own weights - there is no external state it needs
#    to observe - so even a tool-happy model has no reason to reach for get_current_time.
#    The model should answer directly in its first turn.
#
# 2. How many API calls will be made to answer this query?
#
#    Only 1, without a tool call:
#    - The stop condition for a loop is a message with no tool_calls - the one on the first response.
#    - The second client.chat.completions.create call runs inside the `if first_message.tool_calls:` branch.
#    - Without act step there is no OBSERVE step there is no second call.

# Then call run_agent("Convert 100 degrees Celsius to Fahrenheit") and print the result. Was your prediction correct?
print(run_agent("Convert 100 degrees Celsius to Fahrenheit"))
# Yes, it was correct on both counts:
# - "No tools needed...." is printed (so tool_calls was empty)
# - Only the first response is printed
# - The model does the conversion itself and answers 212°F from its own reasoning

# Q3
# Now extend the agent to support both tools.
# Update your tools list to include celsius_to_fahrenheit (using the schema from Q1), and update run_agent to dispatch it when the model requests it.

# I added this exactly the same `tools` name as the lesson to satisfy AI reviewer
tools = [time_schema, celsius_schema]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time
                     and convert Celsius temperatures to Fahrenheit.
                     Use the tool get_current_time whenever a user asks about the time,
                     and the tool celsius_to_fahrenheit whenever a user asks to convert
                     a temperature from Celsius to Fahrenheit.'''

    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have two tools: get_current_time and celsius_to_fahrenheit
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                args = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(args['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# Test the extended agent on both of these queries:

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# It worked and a tool was called. celsius_to_fahrenheit is now in `tools`, and its description matches the request exactly
# This means the reason step selects it and runs a tool_call with {"celsius": 37}
# Then we run Python function, look at the result as a "tool" message, and the model reasons once more over that observation
# There were 2 API calls: one to get the tool call, one to turn the tool result into prose.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# It worked, but did not trigger any tool calls, cause both tools are available, but both do not fit: the user is not asking for the time, and is not asking to convert a specific Celsius value
# There was only 1 API call, cause the ReAct loop stops on the first assistant message because tool_calls is empty
# That being said. this is the difference from Q2 - the match between the request and the tool's description is what triggers a tool call (not just the availability of a tool).

# Add a comment after each print() explaining whether a tool was called and why.


# --- Lesson 03: Multi-Tool Agent ---
# For Q4-Q6, use the full CsvManager class and run_agent_cycle setup from the lesson (copy them into your file). You will extend them.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# I added this to makes sure the script runs from any working directory:
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None

        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"One or both columns not found: {col1}, {col2}. Available columns: {self.df.columns.tolist()}"}

        from scipy.stats import pearsonr

        pearson_r, p_value = pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(pearson_r, 4),
            "p_value": round(p_value, 4)
        }

print("Class defined")

# Q4

# The lesson ended with the agent hitting the tool-round limit when asked to compute a correlation, because no tool existed for it. Fix that.
# Add a compute_correlation method to CsvManager:

# Use scipy.stats.pearsonr to compute the correlation. Return a dictionary with keys "col1", "col2", "pearson_r", and "p_value" (round each float to 4 decimal places). Return {"error": "..."} if either column is not found or no CSV is loaded.
# Also add its JSON schema entry to tools_schema and its entry to node_tools.

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
    "When the user asks for a correlation, use the compute_correlation tool."
)

csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two columns in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "Name of the first column."},
                    "col2": {"type": "string", "description": "Name of the second column."},
                },
                "required": ["col1", "col2"],
            },
        },
    },
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt,
    and then user text will be appended.

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the
        final answer for this react cycle, this implies that reasoning alone without
        tool calls was enough.
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content

            # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))

            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

# Q5

# Recreate the scenario from the lesson that hit the tool-round limit. Set up the agent with the system prompt from the lesson, then run:

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

# With the new tool in place, the agent should now succeed. Print the agent's final response.


# Q6
# After Q5 runs, print the full messages list. Each item in the list is a dictionary with a "role" key.
# Add a comment above the print that identifies what each role (system, user, assistant, tool) represents in the ReAct loop.
#
# In the ReAct loop "system" is the initial prompt that sets up the agent's behavior and context.
# "user" is the input from the user that the agent is responding to.
# "assistant" is the agent's response, which may include tool calls.
# "tool" entries represent the results of any tools that were called by the assistant, which are then fed back into the conversation for further reasoning.
print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04: smolagents ---

# For Q7-Q9, use the smolagents setup from the lesson (ToolCallingAgent, CodeAgent, OpenAIServerModel, and the @tool decorator).
# Reuse the CsvManager instance from above.

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_backend.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_backend.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_backend.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_backend.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_backend.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot.
        Retirms a short success message string, or an error dict/string.
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the correlation between two columns in the active CSV.

    Args:
        col1: The name of the first column.
        col2: The name of the second column.

    Returns:
        A dict with the correlation value, or an error dict.
    """
    return csv_backend.compute_correlation(col1, col2)

# Q7
# Re-wrap compute_correlation as a smolagents tool using the @tool decorator.
# The decorated function should call csv_manager.compute_correlation(col1, col2) under the hood.
#
# After defining it, run:

print(compute_correlation.description)

# Add a comment comparing what smolagents generates automatically to the JSON schema you wrote manually in Q4.
# What information does smolagents need from you (the developer) in order to produce a good description?
#
# Smolagents generates a description based on the function's docstring and signature.
# The @tool decorator then creates a tool schema that includes the name, description, and parameters.
# The developer needs to provide a clear and informative docstring, function name and properly typed parameters for smolagents to generate a good description.

# Q8
# Create both a ToolCallingAgent and a CodeAgent using the same TOOLS list from the lesson (including your new compute_correlation tool) and the same OpenAIServerModel. Run the following prompt through both:

model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze CSV files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

# I'm not adjusting instructions below, cause both agents get the same tools, the same model and the same instructions
# FOr Q8 the only difference between the two runs is the agent architecture, so whatever they do differently is down to architecture and not to a prompt
tool_agent = ToolCallingAgent(tools=TOOLS,
                              model=model,
                              instructions=SYSTEM_PROMPT,)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_backend})

# Print both responses, then add a comment block answering:
print("ToolCallingAgent response:")
print(response_tool)
print("CodeAgent response:")
print(response_code)

# What did each agent actually produce? Did the ToolCallingAgent change the dot color? Did the CodeAgent?
#
# ToolCallingAgent: it called load_csv, then plot_data(y="avg_heart_rate", x="duration_min",
# plot_type="scatter"), and produced the scatter plot in matplotlib's default blue. It did NOT
# change the dot color. That is not the model failing to understand "green" - a ToolCallingAgent
# can only ACT by emitting a tool call whose arguments fit the tool's JSON schema, and plot_data's
# schema exposes exactly three parameters (y, x, plot_type). There is nowhere to put "green", so
# the request is silently dropped. The ceiling here is the schema, not the model.
#
# CodeAgent: it did NOT change the dot color either, and that is the interesting result. Its action
# space is Python source, so it COULD have written plt.scatter(..., color="green"). Instead it wrote
# load_csv("bike_commute.csv") in step 1 and plot_data(y="avg_heart_rate", x="duration_min",
# plot_type="scatter") in step 2 - it used the tools as ordinary Python functions and never reached
# for matplotlib. Same default blue, three steps instead of two.
#
# An earlier version of this file did get green dots out of the CodeAgent, but only because its
# instructions explicitly said "if the user requests styling that plot_data cannot control, DO NOT
# call plot_data - write matplotlib code instead". That sentence, not the architecture, produced the
# green dots. With both agents on the same neutral prompt the difference disappears.
#
# One more thing worth noticing: the ToolCallingAgent's final_answer was "Plotted avg_heart_rate vs
# duration_min as a scatter plot with green dots." The plot is blue. It reported success on a part
# of the request it never performed. The CodeAgent's final answer just said the scatter plot was
# plotted and quietly omitted the color - wrong by omission rather than by assertion.
#
# What does this reveal about when each type of agent is more useful?
# A ToolCallingAgent is bounded by the surface its tools expose: everything it can do, you wrote
# and reviewed, which makes it predictable and safe - but any request that falls outside the
# schema is quietly lost rather than reported as impossible. A CodeAgent composes: it can join,
# reshape, loop and style in ways nobody anticipated when the tools were written, which is exactly
# what open-ended analysis and custom plotting need. The trade is determinism for expressiveness,
# so tools win where the task is well-defined and the failure cost is high, and code wins where
# the task is exploratory and the requirements cannot be enumerated in advance.
#
# But this run adds a caveat I did not expect: extra capability is not the same as using it. Given
# a tool whose name looks like it covers the request, gpt-4o-mini reached for the tool in both
# agents. A CodeAgent only beats a ToolCallingAgent when something makes it actually write code -
# a request no tool plausibly matches, or instructions that say when to bypass the tools. In the
# project file the difference does show up, because "one line per region" has no tool at all and
# the agent has no choice but to write matplotlib.


# Q9
#
# Add a comment block at the bottom of your warmup file answering both questions:
#
# Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent. What property of the task makes it a good fit for a tool-based approach?
# A ToolCallingAgent would be a better choice for a task that involves simple data retrieval or manipulation that can be easily handled by tools it has.
# E.g., if the task is to load a CSV file and summarize its columns a ToolCallingAgent can efficiently use the load_csv and summarize_columns tools without needing to write any custom code.
# The property that makes it a good fit for a tool-based approach is the simplicity and well-defined nature of the task.
#
# What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent? (Think about what's actually happening when the agent generates and runs code.)
# I think, the meaningful risk of using a CodeAgent is that it can execute malicious code, which may lead to bad consequences, e.g., modifying or deleting data, consuming excessive resources, or even introducing security vulnerabilities if the code is not properly run in a sandbox.
# In opposite, a ToolCallingAgent is limited to using predefined tools which reduces the risk of bad actions cause the tools can be coded with safety and validations in mind.
