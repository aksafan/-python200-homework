# --- Supabase Connection ---

# Q1
# In a comment block, answer: what are the two pieces of information supabase-py needs to connect to your project?
# Where do you find them in the Supabase dashboard, and why should they never be hardcoded in a Python script?
# The two pieces of information supabase-py needs to connect to my project are the Supabase Project URL and the Supabase API key (anon key).
# I can find them in my project dashboard by clicking the gear icon (Project Settings) in the left sidebar, then API, then both keys will be listed under "Project API keys" and the URL under "Project URL".

# Q2
# Write a function get_client() that:
#
# Loads your credentials from environment variables using python-dotenv
# Creates and returns a Supabase client
#
# The function should raise a clear error if either environment variable is missing.

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def get_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variable.")

    return create_client(supabase_url, supabase_key)

supabase = get_client()


# Q3
#
# In a comment block, answer: what is Row Level Security (RLS), and why did you disable it on your tables for this course? In what kind of real-world application would you want to keep it enabled?
# Row Level Security (RLS) is the functionality on Supabase to let me define fine-grained access policies — for example, "a user can only read their own rows."
# It is an important production feature, but it adds complexity during development, so that's why I disabled it on my tables for this course.
# In a real-world application where multiple users have access to the same database, I would want to keep RLS enabled to make sure that users can only access their own data and not see or modify other users' data.


# --- supabase-py CRUD ---

# Q1
# Write a function insert_test_record(supabase) that inserts a single row into weather_raw with today's date and plausible values for all four weather columns.
def insert_test_record(supabase):
    from datetime import date

    record = {
        "date": str(date.today()),
        "temperature_2m_max": 30.0,
        "temperature_2m_min": 20.0,
        "precipitation_sum": 5.0,
        "wind_speed_10m_max": 10.0
    }

    response = supabase.table("weather_raw").insert(record).execute()
    print(response)

# insert_test_record(supabase)

# Run it to confirm it works, then add a comment: what would happen if you ran the function twice? How would you change the call to make it safe to run multiple times?
# I ran it and it works, it added data to Supabase.
# If I ran the function twice (and I did that), it showed me an error:
# postgrest.exceptions.APIError: {'message': 'duplicate key value violates unique constraint "weather_raw_pkey"', 'code': '23505', 'hint': None, 'details': 'Key (date)=(2026-09-15) already exists.'}
# So, to make it safe to run multiple times, I would change the call to use upsert instead of insert, which would update the existing row if it already exists.


# Q2
# Write a function get_records_by_date_range(supabase, start, end) that returns all rows from weather_raw where date >= start and date <= end. The function should return the list of row dictionaries.
def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    return response.data

# Test it with a date range that includes the row you inserted in Q1 and print the result.
print(get_records_by_date_range(supabase, '2026-09-14', '2026-09-16'))
# It returned the row I inserted in Q1, so it works:
# [{'date': '2026-09-15', 'temperature_2m_max': 30.0, 'temperature_2m_min': 20.0, 'precipitation_sum': 5.0, 'wind_speed_10m_max': 10.0, 'loaded_at': '2026-09-16T00:20:25.760642+00:00'}]

# Q3
# In a comment block, explain the difference between insert and upsert in supabase-py. Give a concrete example of when you would choose each.
# The difference between insert and upsert in supabase-py is that insert will add a new row to the table, but if a row with the same primary key already exists, it will throw an error (that I saw in console).
# Upsert, on the other hand, will either insert a new row or update the existing row if a conflict occurs on the specified conflict key (in this case, date).

# Then write a function safe_upsert(supabase, records) that upserts a list of records into weather_raw using date as the conflict key and prints the number of rows affected.
def safe_upsert(supabase, records):
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Upserted {len(response.data)} rows into weather_raw.")


# --- Idempotency ---

# Q1
#
# "Idempotency" means that running an operation multiple times produces the same result as running it once. In a comment block, explain why idempotency matters for a data pipeline. Give one concrete example of what goes wrong in a non-idempotent pipeline when the script crashes halfway through and is restarted.
# Idempotency matters for a data pipeline, cause it helps us to not break anything running the pipeline multiple times (for example, due to a crash or a retry), by not producing duplicate or inconsistent data.
# In a non-idempotent pipeline, if the script crashes halfway through and is restarted, it could lead to duplicate records being inserted into the database, which can cause incorrect analysis results or data integrity issues.
# Or it can just crash the script with an exception (like I got in my console).
# An example: if a weather data pipeline inserts daily weather records and crashes after inserting some of the records, restarting the pipeline without idempotency could result in those same records being inserted again, leading to inflated counts or averages in subsequent analyses.
