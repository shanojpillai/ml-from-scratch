import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Get the absolute path of the dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level
file_path = os.path.join(base_dir, "data", "breast_cancer.csv")

# Load dataset
dataset = pd.read_csv(file_path)

# Drop unnecessary columns
X = dataset.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
y = dataset['diagnosis'].map({'M': 1, 'B': -1})  # Convert labels to numerical values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the dataset
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Save processed data
pd.DataFrame(X_train_std).to_csv(os.path.join(base_dir, "data", "X_train_std.csv"), index=False)
pd.DataFrame(X_test_std).to_csv(os.path.join(base_dir, "data", "X_test_std.csv"), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(base_dir, "data", "y_train.csv"), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(base_dir, "data", "y_test.csv"), index=False)

print("✅ Dataset processed and saved!")
