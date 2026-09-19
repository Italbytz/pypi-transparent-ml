import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from imodels import RuleFitClassifier
from scoredrulesets.estimators.rulegp import RuleGPClassifier

from transparent_ml import FuriaClassifier, PartClassifier, IModelsWrapper, ScoredRuleSetsWrapper

def main():
    print("Loading Dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}\n")

    # Define the models we want to unify
    models = {
        "FURIA (Native Transparent-ML)": FuriaClassifier(),
        "PART (Native Transparent-ML)": PartClassifier(),
        "RuleFit (via IModelsWrapper)": IModelsWrapper(RuleFitClassifier(max_rules=5)),
        # ScoredRuleSets wrapper
        "RuleGP (via ScoredRuleSetsWrapper)": ScoredRuleSetsWrapper(RuleGPClassifier())
    }

    for name, model in models.items():
        print(f"=== {name} ===")
        start_time = time.time()
        
        try:
            # 1. Fit on whole dataset to extract rules
            model.fit(X, y)
            print("Extracted Rules:")
            print(model.export_rules())
            
            # 2. Cross Validation
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            elapsed = time.time() - start_time
            print(f"\nAccuracy: {np.mean(scores):.4f} (+/- {np.std(scores)*2:.4f})")
            print(f"Time: {elapsed:.4f}s\n")
        except Exception as e:
            print(f"Failed: {e}\n")

if __name__ == "__main__":
    main()
