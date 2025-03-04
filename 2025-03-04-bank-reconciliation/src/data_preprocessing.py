import pandas as pd
import numpy as np

# Load the cleaned dataset
file_path = "../data/cleaned_banksim.csv"
df = pd.read_csv(file_path)

# Simulate bank records (as a perfect dataset)
bank_records = df.copy()

# Simulate internal records with discrepancies
internal_records = df.copy()

# Introduce missing transactions by randomly removing 5% of transactions
missing_indices = np.random.choice(internal_records.index, size=int(0.05 * len(internal_records)), replace=False)
internal_records.drop(missing_indices, inplace=True)

# Introduce amount variations (rounding errors) in 5% of transactions
variation_indices = np.random.choice(internal_records.index, size=int(0.05 * len(internal_records)), replace=False)
internal_records.loc[variation_indices, 'amount'] *= np.random.uniform(0.98, 1.02, size=len(variation_indices))

# Introduce duplicate transactions (small time shift)
duplicate_indices = np.random.choice(internal_records.index, size=int(0.03 * len(internal_records)), replace=False)
duplicates = internal_records.loc[duplicate_indices].copy()
duplicates['step'] += np.random.randint(1, 3, size=len(duplicates))
internal_records = pd.concat([internal_records, duplicates])

# Save the simulated datasets
bank_records.to_csv("../data/bank_records.csv", index=False)
internal_records.to_csv("../data/internal_records.csv", index=False)

print("Bank records and internal transaction logs generated.")
