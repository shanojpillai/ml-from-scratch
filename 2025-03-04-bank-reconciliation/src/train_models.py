import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load a small sample of the reconciled dataset
file_path = "../data/reconciled_pairs.csv"
df = pd.read_csv(file_path, nrows=50000, dtype={
    'amount_diff': 'float32',
    'amount_pct_diff': 'float32',
    'time_diff': 'int16',
    'merchant_match': 'int8',
    'category_match': 'int8',
    'match': 'int8'
})

# Define features and target
X = df[['amount_diff', 'amount_pct_diff', 'time_diff', 'merchant_match', 'category_match']]
y = df['match']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', probability=True)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print("\nModel Performance:")
print(results_df)

# Save model performance results
results_df.to_csv("../data/model_performance.csv", index=False)
print("Model training completed. Results saved.")