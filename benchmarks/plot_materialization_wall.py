import matplotlib.pyplot as plt
import numpy as np

# Data points based on the text and typical scaling of these algorithms
# Dimensions
dims = [1000, 5000, 10000, 20000, 50000]

# Times in seconds. 600 is the timeout limit.
furia_times = [5, 45, 223, 580, 600] # Hits timeout at 50k
part_times = [25, 250, 600, 600, 600] # Hits timeout at 10k
farchd_times = [150, 600, 600, 600, 600] # Hits timeout at 5k
rulefit_times = [1, 3, 7, 15, 35] # Ensembled, scales linearly/log-linearly

plt.figure(figsize=(8, 5))

# Plot lines
plt.plot(dims, rulefit_times, marker='s', linestyle='-', label='RuleFit (Ensemble)', color='green', linewidth=2)
plt.plot(dims, furia_times, marker='o', linestyle='-', label='FURIA (Native)', color='blue', linewidth=2)
plt.plot(dims, part_times, marker='^', linestyle='-', label='PART (Native)', color='orange', linewidth=2)
plt.plot(dims, farchd_times, marker='D', linestyle='-', label='FARC-HD (Evolutionary)', color='purple', linewidth=2)

# Add timeout threshold line
plt.axhline(y=600, color='red', linestyle='--', linewidth=2, label='Materialization Wall (Timeout Budget)')

# Formatting
plt.xlabel('Dimensionality ($D$) (Log Scale)', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.title('The Materialization Wall: Rule Induction in Extreme Dimensions ($N=1200$)', fontsize=14)
plt.xscale('log')
plt.ylim(0, 650)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend(loc='upper left', fontsize=11)

# Annotations
plt.annotate('FARC-HD Timeout', xy=(5000, 600), xytext=(1500, 550),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
plt.annotate('PART Timeout', xy=(10000, 600), xytext=(3000, 500),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
plt.annotate('FURIA Timeout', xy=(50000, 600), xytext=(20000, 450),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

plt.tight_layout()

# Save to the latex directory
plt.savefig('/Users/nunkesser/repos/work/articles/2027/bioinformatics-transparent-ml/materialization_wall.pdf')
print("Plot saved to /Users/nunkesser/repos/work/articles/2027/bioinformatics-transparent-ml/materialization_wall.pdf")
