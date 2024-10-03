import os
import requests
import time

# Constants for API and match retrieval
BASE_URL = "https://api.opendota.com/api/publicMatches"
MATCH_LIMIT = 100  # Maximum matches per API call
MAX_RANK = "55"  # Max rank to retrieve matches
MAX_MATCHES = 100_000  # Target number of matches to collect
SLEEP_TIME = 0.2  # Delay between API requests
INITIAL_MATCH_ID = 7970066305  # Starting match ID for the query

# Set up the path to save the CSV file in the `data` directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
DATA_DIR = os.path.join(CURRENT_DIR, "../../data")  # Navigate to `data` from `code/datasets`
OUTPUT_FILE = os.path.join(DATA_DIR, "dota2_matches.csv")  # Set the CSV file path in the `data` directory


def fetch_matches(last_match_id, limit=MATCH_LIMIT, max_rank=MAX_RANK):
    """Fetches a batch of matches from OpenDota API based on the given parameters."""
    try:
        response = requests.get(BASE_URL, params={
            "limit": limit,
            "max_rank": max_rank,
            "less_than_match_id": str(last_match_id)
        })
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching matches: {e}")
        return None


def extract_match_data(match):
    """Extracts relevant data from the match information."""
    return {
        "match_id": match["match_id"],
        "winner": match["radiant_win"],  # True if Radiant wins, False if Dire wins
        "radiant_heroes": match["radiant_team"],  # Radiant team heroes (IDs)
        "dire_heroes": match["dire_team"]  # Dire team heroes (IDs)
    }


def save_matches_to_csv(matches, file_path=OUTPUT_FILE, start_id=0):
    """Saves the list of matches to a CSV file without using the csv library."""
    try:
        with open(file_path, "a") as f:
            for idx, match in enumerate(matches, start=start_id):
                # Convert hero IDs from integers to strings and join them
                match_id = match['match_id']
                winner = match['winner']
                radiant_heroes = ",".join(map(str, match['radiant_heroes']))  # Convert int to str
                dire_heroes = ",".join(map(str, match['dire_heroes']))  # Convert int to str
                # Write a single row in CSV format
                f.write(f"{idx},{match_id},{winner},\"{radiant_heroes}\",\"{dire_heroes}\"\n")
    except IOError as e:
        print(f"Error saving matches to CSV file: {e}")


def write_csv_header(file_path=OUTPUT_FILE):
    """Writes the CSV header to the file."""
    try:
        with open(file_path, "w") as f:
            f.write("ID,match_id,winner,radiant_heroes,dire_heroes\n")
    except IOError as e:
        print(f"Error writing CSV header: {e}")


def main():
    """Main function to fetch, process, and store Dota 2 match data."""
    matches = []
    total_matches_collected = 0
    last_match_id = INITIAL_MATCH_ID
    current_id = 0  # Start the ID counter at 0

    # Write CSV header before collecting data
    write_csv_header()

    while total_matches_collected < MAX_MATCHES:
        # Fetch match data from the API
        data = fetch_matches(last_match_id)
        if not data:
            break  # Stop execution if there's an error in fetching data

        # Extract and store relevant match information
        for match in data:
            matches.append(extract_match_data(match))
            total_matches_collected += 1

            if total_matches_collected >= MAX_MATCHES:
                break  # Stop once we've collected the desired number of matches

        print(f"Fetched {len(data)} matches. Total matches collected: {total_matches_collected}")

        # Update the last match ID to ensure pagination
        last_match_id = data[-1]["match_id"]
        time.sleep(SLEEP_TIME)  # To avoid hitting rate limits

        # Save the batch of matches to CSV file, starting with the current ID
        save_matches_to_csv(matches, start_id=current_id)

        # Update the current ID to reflect the number of matches processed
        current_id += len(matches)

        # Clear the list after saving to avoid excessive memory usage
        matches = []

    print(f"Collected a total of {total_matches_collected} matches.")


if __name__ == "__main__":
    main()
