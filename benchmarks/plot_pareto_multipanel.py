import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Consistent model colors & markers
model_styles = {
    'FURIA': {'color': '#1f77b4', 'marker': 'o', 'label': 'FURIA (Native)'},
    'PART': {'color': '#ff7f0e', 'marker': 's', 'label': 'PART (Native)'},
    'RuleGP': {'color': '#2ca02c', 'marker': '^', 'label': 'RuleGP'},
    'RuleFit': {'color': '#d62728', 'marker': 'D', 'label': 'RuleFit'},
    'FARC-HD': {'color': '#7f7f7f', 'marker': 'X', 'label': 'FARC-HD (Native)'}
}

datasets_data = {
    'Breast Cancer ($D=30$)': {
        'FURIA': (11.0, 94.4),
        'PART': (4.2, 92.4),
        'RuleGP': (2.6, 92.7),
        'RuleFit': (57.8, 95.2),
        'FARC-HD': (45.0, 95.6)
    },
    'Colon Cancer ($D=2{,}000$)': {
        'FURIA': (3.2, 74.8),
        'PART': (2.4, 68.9),
        'RuleGP': (3.0, 78.9),
        'RuleFit': (2029.4, 73.3)
    },
    'CNS Embryonal ($D=7{,}129$)': {
        'FURIA': (1.8, 48.4),
        'PART': (2.0, 54.0),
        'RuleGP': (3.2, 52.2),
        'RuleFit': (7160.2, 49.7)
    },
    'Leukemia ($D=7{,}129$)': {
        'FURIA': (3.4, 94.9),
        'PART': (2.4, 81.8),
        'RuleGP': (2.6, 87.8),
        'RuleFit': (7158.6, 94.9)
    },
    'Prostate Cancer ($D=12{,}600$)': {
        'FURIA': (4.8, 81.0),
        'PART': (2.2, 76.1),
        'RuleGP': (2.6, 87.2),
        'RuleFit': (12626.2, 90.1)
    }
}

rank_data = {
    'FURIA': (2.60, 2.50),
    'PART': (1.40, 3.40),
    'RuleGP': (2.00, 2.20),
    'RuleFit': (4.00, 1.90)
}

fig, axes = plt.subplots(3, 2, figsize=(10, 8.2))
axes = axes.flatten()

# Panels (a) to (e): Datasets
panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for idx, (ds_name, data) in enumerate(datasets_data.items()):
    ax = axes[idx]
    ax.set_title(f"{panel_letters[idx]} {ds_name}", fontsize=11, fontweight='bold', pad=8)
    
    # Dynamic y-margin so top and bottom markers have breathing room
    ax.margins(y=0.16)

    for model_name, (rules, f1) in data.items():
        style = model_styles[model_name]
        ax.scatter(rules, f1, color=style['color'], marker=style['marker'], s=90, edgecolors='k', linewidth=0.5, zorder=5)
        
        # Smart annotation offset to avoid overlapping and borders
        offset_x, offset_y = 6, 4
        if 'Breast' in ds_name:
            if model_name == 'FARC-HD':
                offset_x, offset_y = -58, 4
            elif model_name == 'RuleFit':
                offset_x, offset_y = 6, -6
            elif model_name == 'FURIA':
                offset_y = -14
        elif 'Colon' in ds_name:
            if model_name == 'RuleGP':
                offset_x, offset_y = 6, 4
            elif model_name == 'RuleFit':
                offset_x, offset_y = -48, -4
        elif 'CNS' in ds_name:
            if model_name == 'PART':
                offset_x, offset_y = 6, 4
            elif model_name == 'RuleFit':
                offset_x, offset_y = -48, -4
        elif 'Leukemia' in ds_name:
            if model_name == 'RuleFit':
                offset_x, offset_y = -48, -4
            elif model_name == 'FURIA':
                offset_x, offset_y = 6, 4
        elif 'Prostate' in ds_name:
            if model_name == 'RuleFit':
                offset_x, offset_y = -48, -4
            elif model_name == 'RuleGP':
                offset_x, offset_y = 6, 4
            
        ax.annotate(model_name, (rules, f1), xytext=(offset_x, offset_y), textcoords='offset points', fontsize=8.5, fontweight='medium')

    ax.set_xscale('symlog', linthresh=1.0)
    ax.set_xlabel('Model Complexity (# Rules, log scale)', fontsize=9.5)
    ax.set_ylabel('Predictive Quality (Macro F1 %)', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=8.5)

# Panel (f): Global 2D Rank Plot
ax_rank = axes[5]
ax_rank.set_title(f"{panel_letters[5]} Global 2D Rank Overview", fontsize=11, fontweight='bold', pad=8)

# Empirical Pareto Frontier line in rank space (PART -> RuleGP -> RuleFit)
ax_rank.plot([1.40, 2.00, 4.00], [3.40, 2.20, 1.90], linestyle='--', color='#2ca02c', alpha=0.85, linewidth=1.6, zorder=3, label='Pareto Frontier')

for model_name, (rule_rank, f1_rank) in rank_data.items():
    style = model_styles[model_name]
    ax_rank.scatter(rule_rank, f1_rank, color=style['color'], marker=style['marker'], s=120, edgecolors='k', linewidth=0.6, zorder=5)
    
    # Offset annotations
    off_x, off_y = 6, 4
    if model_name == 'FURIA':
        off_x, off_y = 7, -8
    elif model_name == 'RuleGP':
        off_x, off_y = -46, 6
    elif model_name == 'RuleFit':
        off_x, off_y = -46, 6
    elif model_name == 'PART':
        off_x, off_y = 6, 6

    ax_rank.annotate(model_name, (rule_rank, f1_rank), xytext=(off_x, off_y), textcoords='offset points', fontsize=9.5, fontweight='bold')

# Pareto frontier in rank space (Top-Left is optimal: rank 1 in compactness, rank 1 in F1)
ax_rank.set_xlim(0.8, 4.4)
ax_rank.set_ylim(4.3, 0.9) # Inverted so Rank 1 (best) is at the top with generous headroom
ax_rank.set_xlabel('Mean Complexity Rank (1=Most Compact)', fontsize=9.5)
ax_rank.set_ylabel('Mean Predictive Rank (1=Highest F1)', fontsize=9.5)
ax_rank.grid(True, linestyle=':', alpha=0.6)
ax_rank.tick_params(labelsize=8.5)

# Add quadrant arrow / annotation
ax_rank.annotate('Optimal Frontier\n(Compact & Accurate)', xy=(1.5, 1.8), xytext=(2.3, 1.25),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5),
                 fontsize=8.5, fontstyle='italic', ha='center')

# Global legend on top
handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', label=style['label'],
                      markerfacecolor=style['color'], markeredgecolor='k', markersize=9)
           for style in model_styles.values()]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=10, frameon=True, framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../articles/2027/bioinformatics-transparent-ml'))
out_path = os.path.join(out_dir, 'pareto_multipanel.pdf')
plt.savefig(out_path, format='pdf', bbox_inches='tight')
print(f"Saved 6-panel Pareto figure to: {out_path}")
