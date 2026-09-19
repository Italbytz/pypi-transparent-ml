import time
import warnings
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys
import os

# Add local transparent_ml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from transparent_ml import FuriaClassifier, PartClassifier, ScoredRuleSetsWrapper
from imodels import RuleFitClassifier
from scoredrulesets.estimators.rulegp import RuleGPClassifier

warnings.filterwarnings("ignore")

def main():
    print("Loading Arcene from OpenML...")
    arcene = fetch_openml(name='arcene', version=1, as_frame=False, parser='auto')
    X = arcene.data
    if hasattr(arcene.target, 'cat'):
        y = arcene.target.cat.codes.values
    else:
        y = np.array([1 if str(val) == '1' else 0 for val in arcene.target])
    
    print(f"Arcene loaded: N={X.shape[0]}, D={X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        'RuleFit': RuleFitClassifier(random_state=42),
        'FURIA': FuriaClassifier(optimizations=1),
        'PART': PartClassifier(),
        'RuleGP': ScoredRuleSetsWrapper(RuleGPClassifier(population_size=50, max_generations=50, random_state=42))
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- Running {name} ---")
        start_t = time.time()
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - start_t
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro') * 100
            
            rules = 0
            if hasattr(model, 'rules_'):
                rules = len(model.rules_)
            elif hasattr(model, 'rule_set_') and hasattr(model.rule_set_, 'rules'):
                rules = len(model.rule_set_.rules)
            elif hasattr(model, '_model') and hasattr(model._model, 'rules_'):
                rules = len(model._model.rules_)
                
            print(f"{name} -> Time: {train_time:.2f}s, Macro F1: {f1:.1f}%, Rules: {rules}")
            results[name] = {'time': train_time, 'f1': f1, 'rules': rules}
        except Exception as e:
            print(f"{name} -> FAILED: {e}")

    print("\n=== SUMMARY ON ARCENE (D=10,000) ===")
    for name, res in results.items():
        print(f"{name:10s}: Time={res['time']:6.1f}s | F1={res['f1']:5.1f}% | Rules={res['rules']}")

if __name__ == '__main__':
    main()
