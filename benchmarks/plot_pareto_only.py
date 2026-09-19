import os
import matplotlib.pyplot as plt
import numpy as np

valid_models = ['FURIA', 'PART', 'RuleFit', 'RuleGP', 'FARC-HD']

# Values derived from benchmark run
f1_matrix = np.array([
    [94.3, 92.4, 95.0, 92.2, 95.6], # Breast Cancer (FARC-HD performs well)
    [91.6, 81.8, 93.2, 84.6, 0.0],  # Leukemia (FARC-HD timeout)
    [65.2, 68.9, 76.6, 83.9, 0.0]   # Colon (FARC-HD timeout)
])

rules_matrix = np.array([
    [10.8, 4.2, 55.0, 2.0, 45.0],   # Breast Cancer
    [3.2, 2.4, 7155.6, 2.2, 0.0],   # Leukemia
    [3.0, 2.4, 2026.4, 2.6, 0.0]    # Colon
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

plot_path = '/Users/nunkesser/repos/work/articles/2027/bioinformatics-transparent-ml/pareto_tradeoff.pdf'
plt.savefig(plot_path, format='pdf', bbox_inches='tight')
print(f"Saved Pareto plot to: {plot_path}")
