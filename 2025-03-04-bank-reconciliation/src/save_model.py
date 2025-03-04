import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ensure models directory exists
model_dir = "../models/"
os.makedirs(model_dir, exist_ok=True)

# Load reconciled dataset
file_path = "../data/reconciled_pairs.csv"
df = pd.read_csv(file_path, nrows=50000)

# Define features and target
X = df[['amount_diff', 'amount_pct_diff', 'time_diff', 'merchant_match', 'category_match']]
y = df['match']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the best model (Random Forest)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Save the trained model
model_filename = os.path.join(model_dir, "bank_reconciliation_model.pkl")
joblib.dump(best_model, model_filename)
print(f"Model saved successfully as {model_filename}")