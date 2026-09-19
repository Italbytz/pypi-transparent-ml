import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Dimensions
dims = [1000, 5000, 10000, 20000, 50000]

# Times in seconds. 600 is the timeout limit.
farchd_times = [150, 600, 600, 600, 600]     # Hits timeout at 5k
part_times = [25, 250, 600, 600, 600]        # Hits timeout at 10k
furia_times = [5, 45, 223, 580, 600]         # Hits timeout at 50k
rulegp_times = [8.7, 38.6, 75.6, 150.6, 378.5] # Scalable genetic search
rulefit_times = [1, 3, 7, 15, 35]            # Ensembled, scales log-linearly

plt.figure(figsize=(8.5, 5.2))

# Plot lines with consistent colors and markers matching Figure 1
plt.plot(dims, rulefit_times, marker='D', linestyle='-', label='RuleFit (Ensemble)', color='#d62728', linewidth=1.8, markersize=7)
plt.plot(dims, rulegp_times, marker='^', linestyle='-', label='RuleGP (Evolutionary)', color='#2ca02c', linewidth=1.8, markersize=8)
plt.plot(dims, furia_times, marker='o', linestyle='-', label='FURIA (Native Fuzzy)', color='#1f77b4', linewidth=1.8, markersize=7)
plt.plot(dims, part_times, marker='s', linestyle='-', label='PART (Native Decision List)', color='#ff7f0e', linewidth=1.8, markersize=7)
plt.plot(dims, farchd_times, marker='X', linestyle='-', label='FARC-HD (Native Fuzzy)', color='#7f7f7f', linewidth=1.8, markersize=8)

# Add timeout threshold line
plt.axhline(y=600, color='#b22222', linestyle='--', linewidth=1.8, label='Materialization Wall (600s Budget)')

# Formatting
plt.xlabel('Dimensionality ($D$) (Log Scale)', fontsize=11)
plt.ylabel('Training Wall-Clock Time (seconds)', fontsize=11)
plt.title('The Materialization Wall: Rule Induction in Extreme Dimensions ($N=1{,}200$)', fontsize=12, fontweight='bold', pad=10)
plt.xscale('log')
plt.ylim(0, 660)
plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

# Annotations
plt.annotate('FARC-HD Timeout ($D=5{,}000$)', xy=(5000, 600), xytext=(1500, 540),
             arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5), fontsize=8.5, fontweight='medium')
plt.annotate('PART Timeout ($D=10{,}000$)', xy=(10000, 600), xytext=(3500, 480),
             arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5), fontsize=8.5, fontweight='medium')
plt.annotate('FURIA Timeout ($D=50{,}000$)', xy=(50000, 600), xytext=(18000, 420),
             arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5), fontsize=8.5, fontweight='medium')

plt.tight_layout()

# Save to the latex directory
out_path = '/Users/nunkesser/repos/work/articles/2027/bioinformatics-transparent-ml/materialization_wall.pdf'
plt.savefig(out_path, format='pdf', bbox_inches='tight')
print(f"Plot saved to {out_path}")
