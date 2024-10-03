import streamlit as st
import requests

# Set the title of the app
st.title("Dota 2 Match Predictor")


# Define the function to fetch hero data
def fetch_heroes():
    response = requests.get("http://api:8000/heroes/")
    return response.json() if response.status_code == 200 else {}


# Fetch the heroes when the app loads
hero_data = fetch_heroes()

# Create a sidebar for selecting Radiant heroes
st.sidebar.header("Select Radiant Heroes")
radiant_heroes = []
for i in range(5):  # 5 slots for Radiant heroes
    selected_hero = st.sidebar.selectbox(f"Radiant Hero {i + 1}", options=[""] + list(hero_data.values()),
                                         key=f"radiant_{i}")
    radiant_heroes.append(selected_hero)

# Create a sidebar for selecting Dire heroes
st.sidebar.header("Select Dire Heroes")
dire_heroes = []
for i in range(5):  # 5 slots for Dire heroes
    selected_hero = st.sidebar.selectbox(f"Dire Hero {i + 1}", options=[""] + list(hero_data.values()), key=f"dire_{i}")
    dire_heroes.append(selected_hero)

# Button to predict match winner
if st.button("Predict Match Winner"):
    # Get IDs of selected heroes (assuming hero names are unique)
    radiant_ids = [list(hero_data.keys())[list(hero_data.values()).index(hero)] for hero in radiant_heroes if hero]
    dire_ids = [list(hero_data.keys())[list(hero_data.values()).index(hero)] for hero in dire_heroes if hero]

    # Make the prediction request
    if radiant_ids and dire_ids:  # Ensure at least one hero is selected from each team
        payload = {
            "radiant_heroes": radiant_ids,
            "dire_heroes": dire_ids
        }
        response = requests.post("http://api:8000/predict/", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.subheader(f"Winner: {result.get('winner')}")
        else:
            st.error("Failed to get a prediction. Please try again.")
    else:
        st.error("Please select at least one hero from each team.")

# Button to swap heroes
if st.button("Counter Predict Match Winner"):
    # Get IDs of selected heroes (assuming hero names are unique)
    radiant_ids = [list(hero_data.keys())[list(hero_data.values()).index(hero)] for hero in dire_heroes if hero]
    dire_ids = [list(hero_data.keys())[list(hero_data.values()).index(hero)] for hero in radiant_heroes if hero]

    # Make the prediction request
    if radiant_ids and dire_ids:  # Ensure at least one hero is selected from each team
        payload = {
            "radiant_heroes": radiant_ids,
            "dire_heroes": dire_ids
        }
        response = requests.post("http://api:8000/predict/", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.subheader(f"Winner: {result.get('winner')}")
        else:
            st.error("Failed to get a prediction. Please try again.")
    else:
        st.error("Please select at least one hero from each team.")
