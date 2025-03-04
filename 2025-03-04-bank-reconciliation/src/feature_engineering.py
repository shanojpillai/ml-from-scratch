import pandas as pd
import numpy as np
from itertools import product

# Load simulated datasets
bank_records = pd.read_csv("../data/bank_records.csv")
internal_records = pd.read_csv("../data/internal_records.csv")

# Create transaction pairs for reconciliation
pairs = pd.merge(bank_records, internal_records, on=['customer'], suffixes=('_bank', '_internal'))

# Compute matching features
pairs['amount_diff'] = abs(pairs['amount_bank'] - pairs['amount_internal'])
pairs['amount_pct_diff'] = pairs['amount_diff'] / pairs[['amount_bank', 'amount_internal']].max(axis=1)
pairs['time_diff'] = abs(pairs['step_bank'] - pairs['step_internal'])
pairs['merchant_match'] = (pairs['merchant_bank'] == pairs['merchant_internal']).astype(int)
pairs['category_match'] = (pairs['category_bank'] == pairs['category_internal']).astype(int)

# Define matching label (1 = Match, 0 = Mismatch)
match_condition = (
    (pairs['merchant_match'] == 1) & 
    (pairs['category_match'] == 1) & 
    (pairs['amount_pct_diff'] < 0.05) &  # Allow small rounding errors
    (pairs['time_diff'] <= 2)  # Allow time difference of 2 steps
)
pairs['match'] = match_condition.astype(int)

# Save reconciled dataset
pairs.to_csv("../data/reconciled_pairs.csv", index=False)

print("Feature Engineering completed. Reconciled transaction pairs saved.")
