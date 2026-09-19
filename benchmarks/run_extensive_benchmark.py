import os
import sys
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import friedmanchisquare
from imodels import RuleFitClassifier

# Ensure the local transparent_ml module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from transparent_ml import FuriaClassifier, PartClassifier, IModelsWrapper, ScoredRuleSetsWrapper

warnings.filterwarnings("ignore")

def fetch_omics_dataset(name, version=1):
    print(f"Fetching {name} from OpenML...")
    try:
        data = fetch_openml(name=name, version=version, as_frame=True, parser='auto')
        target_series = data.target
        # Binarize target if it's text
        if target_series.dtype.kind in {'U', 'O', 'c'}:
            classes = np.unique(target_series)
            target = np.array([1 if y == classes[1] else 0 for y in target_series])
        else:
            target = target_series.values
            
        # Convert to numpy array and ensure numeric
        X = data.data.select_dtypes(include=[np.number]).values
        feature_names = data.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Impute NaNs with mean
        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            
        return X, target, feature_names
    except Exception as e:
        print(f"Failed to fetch {name}: {e}")
        return None, None, None

def load_datasets():
    datasets = {}
    
    # 1. Breast Cancer (Baseline Bio)
    data = load_breast_cancer()
    datasets['BreastCancer'] = (data.data, data.target, data.feature_names)
    
    # 2. Leukemia (High-Dim Microarray)
    X, y, fnames = fetch_omics_dataset('leukemia')
    if X is not None:
        datasets['Leukemia'] = (X, y, fnames)
        
    # 3. Colon Cancer (High-Dim Microarray)
    X, y, fnames = fetch_omics_dataset('colon')
    if X is not None:
        datasets['Colon'] = (X, y, fnames)
        
    return datasets

def get_models():
    try:
        from scoredrulesets.estimators.rulegp import RuleGPClassifier
        # Keep config minimal to finish in reasonable time, but enough to learn rules
        rulegp_model = RuleGPClassifier(population_size=50, generations=50)
    except ImportError:
        print("WARNING: scoredrulesets not found in PYTHONPATH. RuleGP will fail.")
        rulegp_model = None

    return {
        'FURIA': FuriaClassifier(optimizations=2),
        'PART': PartClassifier(),
        'RuleFit': IModelsWrapper(RuleFitClassifier(max_rules=30)),
        'RuleGP': ScoredRuleSetsWrapper(rulegp_model) if rulegp_model else None
    }

def count_rules(model, feature_names):
    try:
        if hasattr(model, 'rules_'): return len(model.rules_)
        if hasattr(model, 'estimator_') and model.estimator_ is not None:
            if hasattr(model.estimator_, 'to_ruleset'): return len(model.estimator_.to_ruleset().rules)
            if hasattr(model.estimator_, 'get_rules'): return len(model.estimator_.get_rules())
            if hasattr(model.estimator_, 'rules_'): return len(model.estimator_.rules_)
    except Exception:
        pass
    return 0

def run_benchmark():
    datasets = load_datasets()
    models = get_models()
    
    results = {ds: {} for ds in datasets.keys()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for ds_name, (X, y, feature_names) in datasets.items():
        print(f"\n{'='*50}\nEvaluating Dataset: {ds_name} (N={X.shape[0]}, D={X.shape[1]})\n{'='*50}")
        for model_name, model in models.items():
            if model is None: continue
            
            f1_scores = []
            rule_counts = []
            fit_times = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                try:
                    t_start = time.perf_counter()
                    model.fit(X_train, y_train)
                    t_fit = time.perf_counter() - t_start
                    preds = model.predict(X_test)
                    f1_scores.append(f1_score(y_test, preds, average='macro'))
                    rule_counts.append(count_rules(model, feature_names))
                    fit_times.append(t_fit)
                except Exception as e:
                    pass
                    
            if len(f1_scores) > 0:
                results[ds_name][model_name] = {
                    'f1': np.mean(f1_scores) * 100,
                    'rules': np.mean(rule_counts),
                    'time': np.mean(fit_times)
                }
                print(f"  {model_name}: F1={np.mean(f1_scores)*100:.1f}, Rules={np.mean(rule_counts):.1f}, Time={np.mean(fit_times):.1f}s")

    # 1. Statistical Significance (Friedman Test on F1)
    print("\n--- Statistical Significance ---")
    valid_models = [m for m in models.keys() if models[m] is not None]
    f1_matrix = []
    
    for ds in datasets.keys():
        row = []
        for m in valid_models:
            if m in results[ds]:
                row.append(results[ds][m]['f1'])
            else:
                row.append(0.0)
        f1_matrix.append(row)
        
    f1_matrix = np.array(f1_matrix)
    if f1_matrix.shape[0] > 0 and f1_matrix.shape[1] >= 3:
        stat, p = friedmanchisquare(*[f1_matrix[:, i] for i in range(f1_matrix.shape[1])])
        print(f"Friedman Test p-value: {p:.4f} (Stat: {stat:.4f})")
    
    # 2. Pareto Plot Generation
    print("\n--- Generating Pareto Plot ---")
    avg_f1 = np.mean(f1_matrix, axis=0)
    
    rule_matrix = []
    for ds in datasets.keys():
        row = []
        for m in valid_models:
            if m in results[ds]:
                row.append(results[ds][m]['rules'])
            else:
                row.append(0.0)
        rule_matrix.append(row)
    avg_rules = np.mean(rule_matrix, axis=0)
    
    plt.figure(figsize=(8, 6))
    for i, m in enumerate(valid_models):
        plt.scatter(avg_rules[i], avg_f1[i], label=m, s=150)
        plt.annotate(m, (avg_rules[i], avg_f1[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.xlabel('Model Complexity (Average Number of Rules)')
    plt.ylabel('Predictive Quality (Average Macro F1 %)')
    plt.title('Pareto Trade-off: Accuracy vs. Interpretability')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../articles/2027/bioinformatics-transparent-ml/pareto_tradeoff.pdf'))
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved Pareto plot to: {plot_path}")
    
    # Dump LaTeX table rows for easy copy-pasting (with timing)
    print("\n--- LaTeX Table Output (with Timing) ---")
    for m in valid_models:
        row_str = m
        for ds in datasets.keys():
            res = results[ds].get(m, None)
            if res:
                row_str += f" & {res['f1']:.1f} & {res['rules']:.1f} & {res['time']:.1f}"
            else:
                row_str += " & - & - & -"
        row_str += " \\\\"
        print(row_str)

if __name__ == "__main__":
    run_benchmark()
