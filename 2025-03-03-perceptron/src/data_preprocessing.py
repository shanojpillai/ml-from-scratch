import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "../data/breast_cancer.csv"  # Adjust path if necessary
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
pd.DataFrame(X_train_std).to_csv("../data/X_train_std.csv", index=False)
pd.DataFrame(X_test_std).to_csv("../data/X_test_std.csv", index=False)
pd.DataFrame(y_train).to_csv("../data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("../data/y_test.csv", index=False)

print("✅ Dataset processed and saved!")
