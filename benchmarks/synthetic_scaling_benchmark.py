import time
import warnings
import signal
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imodels import RuleFitClassifier
import sys
import os

# Ensure the local transparent_ml module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from transparent_ml import FuriaClassifier, PartClassifier, FarcHDClassifier

warnings.filterwarnings("ignore")

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout reached")

def main():
    budget = 600
    signal.signal(signal.SIGALRM, timeout_handler)
    
    dims = [10000, 50000, 100000, 250000, 500000]
    results = {}

    for p in dims:
        print(f"\n--- Testing synthetic dataset: N=1200, D={p} ---")
        try:
            X, y = make_classification(
                n_samples=1200, 
                n_features=p, 
                n_informative=6, 
                n_redundant=0, 
                n_repeated=0, 
                n_classes=2, 
                flip_y=0.05, 
                random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        except MemoryError:
            print(f"MemoryError during data generation for D={p}. Aborting further scaling.")
            break

        models = {
            'FURIA': FuriaClassifier(optimizations=1),
            'PART': PartClassifier(),
            'FARC-HD': FarcHDClassifier(time_limit_sec=budget),
            'RuleFit': RuleFitClassifier()
        }

        results[p] = {}

        for name, model in models.items():
            print(f"  Training {name} (Budget: {budget}s)...", end="", flush=True)
            start_time = time.time()
            signal.alarm(budget)
            try:
                model.fit(X_train, y_train)
                signal.alarm(0) # Cancel alarm
                fit_time = time.time() - start_time
                acc = model.score(X_test, y_test)
                rules = len(model.rules_) if hasattr(model, 'rules_') else 0
                print(f" DONE: {fit_time:.1f}s | Acc: {acc:.3f} | Rules: {rules}")
                results[p][name] = "OK"
            except TimeoutException:
                print(" TIMEOUT")
                results[p][name] = "TIMEOUT"
            except MemoryError:
                signal.alarm(0)
                print(" MEMORY ERROR")
                results[p][name] = "OOM"
            except Exception as e:
                signal.alarm(0)
                print(f" FAILED ({e})")
                results[p][name] = "FAILED"

        # Check if all timed out or OOM
        statuses = [results[p][m] for m in models]
        if all(s in ["TIMEOUT", "OOM", "FAILED"] for s in statuses):
            print("All models failed or timed out at this scale. Stopping early.")
            break

if __name__ == '__main__':
    main()
