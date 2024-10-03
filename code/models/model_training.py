import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import joblib

# Set up the path for CSV file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
DATA_DIR = os.path.join(CURRENT_DIR, "../../data")  # Navigate to `data`
INPUT_FILE = os.path.join(DATA_DIR, "dota2_matches.csv")  # Set the CSV file path in the `data` directory
# Load the data
df = pd.read_csv(INPUT_FILE)

# Define hero IDs
hero_ids = {1: 'Anti-Mage', 2: 'Axe', 3: 'Bane', 4: 'Bloodseeker', 5: 'Crystal Maiden', 6: 'Drow Ranger',
            7: 'Earthshaker', 8: 'Juggernaut', 9: 'Mirana', 10: 'Morphling', 11: 'Shadow Fiend', 12: 'Phantom Lancer',
            13: 'Puck', 14: 'Pudge', 15: 'Razor', 16: 'Sand King', 17: 'Storm Spirit', 18: 'Sven', 19: 'Tiny',
            20: 'Vengeful Spirit', 21: 'Windranger', 22: 'Zeus', 23: 'Kunkka', 25: 'Lina', 26: 'Lion',
            27: 'Shadow Shaman', 28: 'Slardar', 29: 'Tidehunter', 30: 'Witch Doctor', 31: 'Lich', 32: 'Riki',
            33: 'Enigma', 34: 'Tinker', 35: 'Sniper', 36: 'Necrophos', 37: 'Warlock', 38: 'Beastmaster',
            39: 'Queen of Pain', 40: 'Venomancer', 41: 'Faceless Void', 42: 'Wraith King', 43: 'Death Prophet',
            44: 'Phantom Assassin', 45: 'Pugna', 46: 'Templar Assassin', 47: 'Viper', 48: 'Luna', 49: 'Dragon Knight',
            50: 'Dazzle', 51: 'Clockwerk', 52: 'Leshrac', 53: "Nature's Prophet", 54: 'Lifestealer', 55: 'Dark Seer',
            56: 'Clinkz', 57: 'Omniknight', 58: 'Enchantress', 59: 'Huskar', 60: 'Night Stalker', 61: 'Broodmother',
            62: 'Bounty Hunter', 63: 'Weaver', 64: 'Jakiro', 65: 'Batrider', 66: 'Chen', 67: 'Spectre',
            68: 'Ancient Apparition', 69: 'Doom', 70: 'Ursa', 71: 'Spirit Breaker', 72: 'Gyrocopter', 73: 'Alchemist',
            74: 'Invoker', 75: 'Silencer', 76: 'Outworld Destroyer', 77: 'Lycan', 78: 'Brewmaster', 79: 'Shadow Demon',
            80: 'Lone Druid', 81: 'Chaos Knight', 82: 'Meepo', 83: 'Treant Protector', 84: 'Ogre Magi', 85: 'Undying',
            86: 'Rubick', 87: 'Disruptor', 88: 'Nyx Assassin', 89: 'Naga Siren', 90: 'Keeper of the Light', 91: 'Io',
            92: 'Visage', 93: 'Slark', 94: 'Medusa', 95: 'Troll Warlord', 96: 'Centaur Warrunner', 97: 'Magnus',
            98: 'Timbersaw', 99: 'Bristleback', 100: 'Tusk', 101: 'Skywrath Mage', 102: 'Abaddon', 103: 'Elder Titan',
            104: 'Legion Commander', 105: 'Techies', 106: 'Ember Spirit', 107: 'Earth Spirit', 108: 'Underlord',
            109: 'Terrorblade', 110: 'Phoenix', 111: 'Oracle', 112: 'Winter Wyvern', 113: 'Arc Warden',
            114: 'Monkey King', 119: 'Dark Willow', 120: 'Pangolier', 121: 'Grimstroke', 123: 'Hoodwink',
            126: 'Void Spirit', 128: 'Snapfire', 129: 'Mars', 131: 'Ringmaster', 135: 'Dawnbreaker', 136: 'Marci',
            137: 'Primal Beast', 138: 'Muerta'}

# Create binary columns for each hero
for hero_id, hero_name in hero_ids.items():
    radiant_col = f'radi_{hero_id}'
    dire_col = f'dire_{hero_id}'

    # Check if hero ID is in radiant heroes
    df[radiant_col] = df['radiant_heroes'].apply(lambda x: 1 if str(hero_id) in x.split(', ') else 0)

    # Check if hero ID is in dire heroes
    df[dire_col] = df['dire_heroes'].apply(lambda x: 1 if str(hero_id) in x.split(', ') else 0)

# Prepare features and target
X = df[[f'radi_{hero_id}' for hero_id in hero_ids.keys()] +
       [f'dire_{hero_id}' for hero_id in hero_ids.keys()]]
Y = df['radiant_win'].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Train LightGBM model
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

lgb_params = {
    'learning_rate': 0.003,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'sub_feature': 0.5,
    'num_leaves': 5,
    'min_data': 100,
    'max_depth': 10
}

# Train the model with early stopping in valid_sets
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100,
                      valid_sets=[lgb_eval], valid_names=['eval'],
                      callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation(5)])

# Make predictions and evaluate
lgb_predictions = lgb_model.predict(X_test)
lgb_predictions = [1 if pred >= 0.5 else 0 for pred in lgb_predictions]

lgb_confusion = confusion_matrix(y_test, lgb_predictions)
lgb_f1_score = f1_score(y_test, lgb_predictions)

print(f'LightGBM F1 Score: {lgb_f1_score * 100:.2f}%')

# Plot confusion matrix
sns.heatmap(lgb_confusion, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Radiant Win', 'Radiant Loss'],
            yticklabels=['Radiant Win', 'Radiant Loss'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for LightGBM')
plt.show()

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_confusion = confusion_matrix(y_test, rf_predictions)
rf_f1_score = f1_score(y_test, rf_predictions)

print(f'Random Forest F1 Score: {rf_f1_score * 100:.2f}%')

# Plot Random Forest confusion matrix
sns.heatmap(rf_confusion, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Radiant Win', 'Radiant Loss'],
            yticklabels=['Radiant Win', 'Radiant Loss'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Random Forest')
plt.show()

# Train k-NN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

knn_confusion = confusion_matrix(y_test, knn_predictions)
knn_f1_score = f1_score(y_test, knn_predictions)

print(f'KNN F1 Score: {knn_f1_score * 100:.2f}%')

# Set up the path to save the pkl file in the `models` directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
DATA_DIR = os.path.join(CURRENT_DIR, "../../models")  # Navigate to `models`
OUTPUT_FILE = os.path.join(DATA_DIR, "RandomForestDota.pkl")  # Set the pkl file path in the `models` directory
# Save the Random Forest model
joblib.dump(rf_model, OUTPUT_FILE)
