import os
import matplotlib.pyplot as plt
import numpy as np

valid_models = ['FURIA', 'PART', 'RuleFit', 'RuleGP', 'FARC-HD']

# Values derived from 5-dataset benchmark run (5-fold CV)
f1_matrix = np.array([
    [94.4, 92.4, 95.2, 92.7, 95.6], # Breast Cancer (D=30)
    [74.8, 68.9, 73.3, 78.9, 0.0],  # Colon (D=2,000, FARC-HD timeout)
    [48.4, 54.0, 49.7, 52.2, 0.0],  # CNS (D=7,129, FARC-HD timeout)
    [94.9, 81.8, 94.9, 87.8, 0.0],  # Leukemia (D=7,129, FARC-HD timeout)
    [81.0, 76.1, 90.1, 87.2, 0.0]   # Prostate (D=12,600, FARC-HD timeout)
])

rules_matrix = np.array([
    [11.0, 4.2, 57.8, 2.6, 45.0],   # Breast Cancer
    [3.2, 2.4, 2029.4, 3.0, 0.0],   # Colon
    [1.8, 2.0, 7160.2, 3.2, 0.0],   # CNS
    [3.4, 2.4, 7158.6, 2.6, 0.0],   # Leukemia
    [4.8, 2.2, 12626.2, 2.6, 0.0]   # Prostate
])

avg_f1 = np.mean(f1_matrix, axis=0)
avg_rules = np.mean(rules_matrix, axis=0)

plt.figure(figsize=(8, 6))
for i, m in enumerate(valid_models):
    plt.scatter(avg_rules[i], avg_f1[i], label=m, s=150)
    plt.annotate(m, (avg_rules[i], avg_f1[i]), xytext=(5, 5), textcoords='offset points')
    
plt.xlabel('Model Complexity (Average Number of Rules)')
plt.ylabel('Predictive Quality (Average Macro F1 %)')
plt.title('Pareto Trade-off: Accuracy vs. Interpretability on Omics Datasets')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xscale('symlog') # Use log scale because RuleFit is >3000 rules

plot_path = 'benchmarks/results/pareto_tradeoff.pdf'
plt.savefig(plot_path, format='pdf', bbox_inches='tight')
print(f"Saved Pareto plot to: {plot_path}")
