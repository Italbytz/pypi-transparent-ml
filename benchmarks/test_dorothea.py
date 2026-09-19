import time
import warnings
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
import sys
import os

# Ensure the local transparent_ml module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from transparent_ml import FuriaClassifier, PartClassifier

warnings.filterwarnings("ignore")

def main():
    print("Fetching Arcene from OpenML...")
    try:
        dorothea = fetch_openml(name='arcene', version=1, as_frame=False, parser='auto')
        X = dorothea.data
        if hasattr(dorothea.target, 'cat'):
            y = dorothea.target.cat.codes.values
        else:
            y = np.array([1 if str(val) == '1' else 0 for val in dorothea.target])
        print(f"Arcene Loaded: N={X.shape[0]}, D={X.shape[1]}")
    except Exception as e:
        print(f"Failed to fetch Arcene: {e}")
        return

    # Use a small subset to test memory and speed first (e.g. 1 fold, or just fit on full)
    from sklearn.model_selection import train_test_split
    from imodels import RuleFitClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'FURIA': FuriaClassifier(optimizations=1),
        'PART': PartClassifier(),
        'RuleFit': RuleFitClassifier()
    }

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            fit_time = time.time() - start_time
            acc = model.score(X_test, y_test)
            
            rules = 0
            if hasattr(model, 'rules_'):
                rules = len(model.rules_)
                
            print(f"  {name} -> Acc: {acc:.3f}, Time: {fit_time:.2f}s, Rules: {rules}")
        except Exception as e:
            print(f"  {name} -> FAILED: {e}")

if __name__ == '__main__':
    main()
