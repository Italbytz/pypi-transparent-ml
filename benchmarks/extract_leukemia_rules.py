from sklearn.datasets import fetch_openml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from transparent_ml import FuriaClassifier

print("Fetching Leukemia dataset...")
data = fetch_openml(data_id=1104, as_frame=True)
X, y = data.data, data.target

# Ensure all columns are numeric
import numpy as np
X = X.select_dtypes(include=[np.number])

print("Training FURIA on Leukemia dataset...")
model = FuriaClassifier()
model.fit(X, y)

print("\n--- FURIA Rules for Leukemia ---")
for r in model.rules_:
    print(r)
print(f"Total Rules: {len(model.rules_)}")
