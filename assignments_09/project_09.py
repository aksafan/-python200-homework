import os

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client

# Part 2: Project — Extract + Load Pipeline

# Build project_09.py, a script that implements a complete Extract + Load pipeline: it fetches 2023 daily weather data from the Open-Meteo API for a city of your choice and loads it into your Supabase weather_raw table.
#
# This is the same data your Week 4 classifier was trained on. In later weeks, you will use these rows as the input to the transform step — so make sure your column values match what the model expects.

load_dotenv()

def get_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variable.")

    return create_client(supabase_url, supabase_key)


# --- Step 1: Extract ---

# Call the Open-Meteo historical archive API to retrieve daily weather data for your chosen city for the full year 2023 (start: 2023-01-01, end: 2023-12-31). Use these four daily variables:
#
# temperature_2m_max
# temperature_2m_min
# precipitation_sum
# wind_speed_10m_max
#
# Use response.raise_for_status() to catch errors early. Print a summary of the response once it arrives.

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

def extract():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 30.2672,  # Austin, TX
        "longitude": -97.7431,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": DAILY_VARIABLES,
        "timezone": "America/Chicago",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    daily = payload["daily"]
    print("A summary of the response")
    print(f"Status: {response.status_code}, elapsed: {response.elapsed.total_seconds():.2f}s")
    print(f"Location: lat {payload['latitude']}, lon {payload['longitude']}, timezone {payload['timezone']}")
    print(f"Daily columns: {list(daily.keys())}")
    print(f"Days returned: {len(daily['time'])} ({daily['time'][0]} .. {daily['time'][-1]})")

    return daily


# --- Step 2: Transform ---

# Convert the API response from columnar format into a list of row dictionaries. Each dictionary should have keys that exactly match the column names in weather_raw.
#
# Print the first and last record to confirm the transformation looks correct.

def transform(daily):
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df = df[["date"] + DAILY_VARIABLES]

    # Fixing NaN here, cause it can work only with None (will transform to SQL NULL)
    df = df.astype(object).where(pd.notna(df), None)
    records = df.to_dict("records")

    print("\nPrinting the first and last record to confirm the transformation looks correct")
    print(f"Records: {len(records)}")
    print(f"First: {records[0]}")
    print(f"Last:  {records[-1]}")

    return records

# Add a comment: how many records do you expect for a full year, and how many did you get? If the numbers differ, what might explain the discrepancy?

# I checked 2023 and it is not a leap year, so I'd expect 365 records for 365 days - one row per day.
# I got exactly 365 with First: {'date': '2023-01-01'}, and Last:  {'date': '2023-12-31'}.
# If the numbers differed, I assume it could be cause of:
# - Time zone issues tht shifted the start or end date by a day
# - The archive site issues, maybe on their DB side
# - The station data having gaps (e.g. with null measurements), so some days come back with null measurements at all, but is unlikely for a full year of daily data.


# --- Step 3: Load ---

# Upsert all records into weather_raw. Print a confirmation message showing how many rows were upserted.
# Run the script a second time and confirm the row count in weather_raw does not change.

TABLE = "weather_raw"
CHUNK_SIZE = 200

def load(supabase, records):
    print("\nUpserting all records into weather_raw")

    upserted = 0
    for start in range(0, len(records), CHUNK_SIZE):
        chunk = records[start:start + CHUNK_SIZE]
        response = supabase.table(TABLE).upsert(chunk, on_conflict="date").execute()
        upserted += len(response.data)

    print(f"Upserted {upserted} rows into {TABLE}.")

    return upserted

# Add a comment: what does this tell you about idempotency?

# So, I ran the script multiple times and every time the row count was the same (365).
# Idempotency worked here, cause every time a `date` was the primary key and I upserted on it, and then all rest of runs matched every existing row and overwrote it instead of inserting a duplicate.
# This is what idempotency means in practice.
# In other words, the state of the table after multiple runs is identical to the state after the first run,
# This will make sure that a crashed pipeline can can be re-run from the top without any cleanup step or corrupting data.
# One thing that I noticed is that `loaded_at` did not change on the second run.
# I think this is cause the upsert does an UPDATE of only the columns I send, and I didn't send `loaded_at` column.


# --- Step 4: Verify ---

# After upserting, run a verification query that:
#
# Prints the total number of rows in weather_raw
# Prints the earliest and latest dates in the table
# Prints the row for 2023-07-04 (or the nearest date if that date is missing)
#
# You can also verify directly in the Supabase Table Editor — take a screenshot for the video below.

def verify(supabase, target_date="2023-07-04"):
    print("\nRunning a verification query")

    total = supabase.table(TABLE).select("date", count="exact").limit(1).execute().count
    print(f"Total rows: {total}")

    earliest = supabase.table(TABLE).select("date").order("date").limit(1).execute().data
    latest = supabase.table(TABLE).select("date").order("date", desc=True).limit(1).execute().data
    print(f"Earliest date: {earliest[0]['date']}")
    print(f"Latest date: {latest[0]['date']}")

    row = supabase.table(TABLE).select("*").eq("date", target_date).execute().data
    if row:
        print(f"Row for {target_date}: {row[0]}")
    else:
        # Fall back to the nearest available date on either side of the target.
        before = supabase.table(TABLE).select("*").lt("date", target_date).order("date", desc=True).limit(1).execute().data
        after = supabase.table(TABLE).select("*").gt("date", target_date).order("date").limit(1).execute().data
        nearest = min(
            before + after,
            key=lambda r: abs(pd.Timestamp(r["date"]) - pd.Timestamp(target_date)),
        )
        print(f"No row for {target_date}, nearest is {nearest['date']}: {nearest}")


def main():
    supabase = get_client()

    daily = extract()
    records = transform(daily)
    load(supabase, records)
    verify(supabase)


if __name__ == "__main__":
    main()


# Video
#
# Record a short video (target: 3 minutes, max: 5). Show:
#
# The script running in your terminal with no errors
# The weather_raw table in your Supabase dashboard with rows visible
# Your verification output printed to the terminal
#
# Paste the video link in a comment at the top of project_09.py.

# https://youtu.be/AkcM_xNNFzE
