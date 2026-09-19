import warnings
import sys
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from imodels import RuleFitClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from transparent_ml import FuriaClassifier, PartClassifier

warnings.filterwarnings("ignore")

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("--- FURIA Rules ---")
    furia = FuriaClassifier(optimizations=1)
    furia.fit(X_train, y_train)
    print(furia.export_rules())

    print("\n--- PART Rules ---")
    part = PartClassifier()
    part.fit(X_train, y_train)
    print(part.export_rules())

    print("\n--- RuleFit Rules ---")
    rulefit = RuleFitClassifier(max_rules=10) # Limit for readability
    rulefit.fit(X_train, y_train, feature_names=feature_names)
    print(rulefit.rules_)

if __name__ == '__main__':
    main()
