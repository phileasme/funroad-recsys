import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set theme for visualizations
sns.set_theme(style="whitegrid")

# Parse the provided results
metrics = {
    "ndcg": {
        "1": {"gumroad": 0.0652, "endpoint": 0.1642},
        "3": {"gumroad": 0.0768, "endpoint": 0.2088},
        "5": {"gumroad": 0.0991, "endpoint": 0.2682},
        "10": {"gumroad": 0.1467, "endpoint": 0.3474}
    },
    "weighted_precision": {
        "1": {"gumroad": 0.0756, "endpoint": 0.1787},
        "3": {"gumroad": 0.0760, "endpoint": 0.1923},
        "5": {"gumroad": 0.0887, "endpoint": 0.2188},
        "10": {"gumroad": 0.0938, "endpoint": 0.1916}
    },
    "mrr": {"gumroad": 0.5053, "endpoint": 0.5037},
    "first_relevant_position": {"gumroad": 2.13, "endpoint": 2.09, "improvement": 0.04},
    "avg_relevant_rank": {"gumroad": 24.83, "endpoint": 11.34, "improvement": 13.48}
}

# 1. Create comprehensive dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# nDCG plot
ax = axes[0, 0]
k_values = sorted([int(k) for k in metrics['ndcg'].keys()])
gumroad_ndcg = [metrics['ndcg'][str(k)]['gumroad'] for k in k_values]
endpoint_ndcg = [metrics['ndcg'][str(k)]['endpoint'] for k in k_values]

ax.plot(k_values, gumroad_ndcg, 'o-', label='Gumroad', color='#1f77b4', linewidth=2, markersize=8)
ax.plot(k_values, endpoint_ndcg, 'o-', label='Custom Endpoint', color='#2ca02c', linewidth=2, markersize=8)
ax.set_title('nDCG@k Comparison', fontsize=16)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('nDCG', fontsize=14)
ax.set_xticks(k_values)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add value labels
for i, (gval, eval) in enumerate(zip(gumroad_ndcg, endpoint_ndcg)):
    ax.annotate(f'{gval:.4f}', (k_values[i], gval), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=10)
    ax.annotate(f'{eval:.4f}', (k_values[i], eval), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=10)

# Weighted Precision plot
ax = axes[0, 1]
gumroad_wp = [metrics['weighted_precision'][str(k)]['gumroad'] for k in k_values]
endpoint_wp = [metrics['weighted_precision'][str(k)]['endpoint'] for k in k_values]

ax.plot(k_values, gumroad_wp, 'o-', label='Gumroad', color='#1f77b4', linewidth=2, markersize=8)
ax.plot(k_values, endpoint_wp, 'o-', label='Custom Endpoint', color='#2ca02c', linewidth=2, markersize=8)
ax.set_title('Weighted Precision@k Comparison', fontsize=16)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('Weighted Precision', fontsize=14)
ax.set_xticks(k_values)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add value labels
for i, (gval, eval) in enumerate(zip(gumroad_wp, endpoint_wp)):
    ax.annotate(f'{gval:.4f}', (k_values[i], gval), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=10)
    ax.annotate(f'{eval:.4f}', (k_values[i], eval), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=10)

# MRR plot
ax = axes[1, 0]
mrr_values = [metrics['mrr']['gumroad'], metrics['mrr']['endpoint']]
x = ['Gumroad', 'Custom Endpoint']

bars = ax.bar(x, mrr_values, color=['#1f77b4', '#2ca02c'], width=0.6)
ax.set_title('Mean Reciprocal Rank (MRR)', fontsize=16)
ax.set_ylabel('MRR', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(mrr_values) * 1.15)  # Add some headroom for labels

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=12)

# Position metrics plot
ax = axes[1, 1]

position_metrics = ['First Relevant Position', 'Avg Relevant Rank']
gumroad_positions = [metrics['first_relevant_position']['gumroad'],
                    metrics['avg_relevant_rank']['gumroad']]
endpoint_positions = [metrics['first_relevant_position']['endpoint'],
                     metrics['avg_relevant_rank']['endpoint']]

x = np.arange(len(position_metrics))
width = 0.35

ax.bar(x - width/2, gumroad_positions, width, label='Gumroad', color='#1f77b4')
ax.bar(x + width/2, endpoint_positions, width, label='Custom Endpoint', color='#2ca02c')
ax.set_title('Position Metrics (Lower is Better)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(position_metrics, fontsize=12)
ax.set_ylabel('Position', fontsize=14)
ax.legend(fontsize=12)

# Add value labels
for i, v in enumerate(gumroad_positions):
    ax.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center', fontsize=12)
for i, v in enumerate(endpoint_positions):
    ax.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center', fontsize=12)

# Set y-scale for better visualization since avg rank is much larger
if metrics['avg_relevant_rank']['gumroad'] > 15:
    ax.set_yscale('symlog')  # Symmetric log scale works well for this case
    ax.set_ylim(0, metrics['avg_relevant_rank']['gumroad'] * 1.2)

plt.tight_layout()
plt.savefig('search_metrics_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create improvement visualization
plt.figure(figsize=(14, 8))

# Calculate improvement percentages
improvements = []
labels = []

# nDCG improvements
for k in k_values:
    gumroad_val = metrics['ndcg'][str(k)]['gumroad']
    endpoint_val = metrics['ndcg'][str(k)]['endpoint']
    improvement = ((endpoint_val - gumroad_val) / gumroad_val) * 100
    improvements.append(improvement)
    labels.append(f'nDCG@{k}')

# Weighted Precision improvements
for k in k_values:
    gumroad_val = metrics['weighted_precision'][str(k)]['gumroad']
    endpoint_val = metrics['weighted_precision'][str(k)]['endpoint']
    improvement = ((endpoint_val - gumroad_val) / gumroad_val) * 100
    improvements.append(improvement)
    labels.append(f'WP@{k}')

# MRR improvement
gumroad_val = metrics['mrr']['gumroad']
endpoint_val = metrics['mrr']['endpoint']
improvement = ((endpoint_val - gumroad_val) / gumroad_val) * 100
improvements.append(improvement)
labels.append('MRR')

# Position metric improvements (lower is better, so reversed)
gumroad_val = metrics['first_relevant_position']['gumroad']
endpoint_val = metrics['first_relevant_position']['endpoint']
# For position metrics, improvement is the percentage reduction
if gumroad_val > 0:
    improvement = ((gumroad_val - endpoint_val) / gumroad_val) * 100
else:
    improvement = 0
improvements.append(improvement)
labels.append('First Relevant Pos')

gumroad_val = metrics['avg_relevant_rank']['gumroad']
endpoint_val = metrics['avg_relevant_rank']['endpoint']
if gumroad_val > 0:
    improvement = ((gumroad_val - endpoint_val) / gumroad_val) * 100
else:
    improvement = 0
improvements.append(improvement)
labels.append('Avg Relevant Rank')

# Sort by improvement for better visualization
sorted_indices = np.argsort(improvements)
sorted_improvements = [improvements[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

y_pos = np.arange(len(sorted_labels))
colors = ['#2ca02c' if x > 0 else '#d62728' for x in sorted_improvements]

bars = plt.barh(y_pos, sorted_improvements, color=colors)
plt.yticks(y_pos, sorted_labels, fontsize=12)
plt.xlabel('Relative Improvement (%)', fontsize=14)
plt.title('Custom Endpoint Performance Improvements', fontsize=18)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    label_x = width + 1 if width > 0 else width - 6
    plt.text(label_x, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', va='center',
             color='black',
             ha='left' if width > 0 else 'right',
             fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('search_improvement_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Create a visual summary with absolute values and improvements
fig, ax = plt.subplots(figsize=(14, 10))

# Combine all metrics for a summary table
summary_metrics = []
summary_labels = []
gumroad_values = []
endpoint_values = []
improvements_values = []
improvement_percentages = []

# Add nDCG metrics
for k in k_values:
    summary_labels.append(f"nDCG@{k}")
    gumroad_values.append(metrics['ndcg'][str(k)]['gumroad'])
    endpoint_values.append(metrics['ndcg'][str(k)]['endpoint'])
    improvement = endpoint_values[-1] - gumroad_values[-1]
    improvements_values.append(improvement)
    pct_improvement = (improvement / gumroad_values[-1]) * 100 if gumroad_values[-1] > 0 else 0
    improvement_percentages.append(pct_improvement)

# Add Weighted Precision metrics
for k in k_values:
    summary_labels.append(f"Weighted Precision@{k}")
    gumroad_values.append(metrics['weighted_precision'][str(k)]['gumroad'])
    endpoint_values.append(metrics['weighted_precision'][str(k)]['endpoint'])
    improvement = endpoint_values[-1] - gumroad_values[-1]
    improvements_values.append(improvement)
    pct_improvement = (improvement / gumroad_values[-1]) * 100 if gumroad_values[-1] > 0 else 0
    improvement_percentages.append(pct_improvement)

# Add MRR
summary_labels.append("MRR")
gumroad_values.append(metrics['mrr']['gumroad'])
endpoint_values.append(metrics['mrr']['endpoint'])
improvement = endpoint_values[-1] - gumroad_values[-1]
improvements_values.append(improvement)
pct_improvement = (improvement / gumroad_values[-1]) * 100 if gumroad_values[-1] > 0 else 0
improvement_percentages.append(pct_improvement)

# Add Position metrics (lower is better, so reverse the sign of improvement)
summary_labels.append("First Relevant Position")
gumroad_values.append(metrics['first_relevant_position']['gumroad'])
endpoint_values.append(metrics['first_relevant_position']['endpoint'])
improvement = gumroad_values[-1] - endpoint_values[-1]  # Note: reversed because lower is better
improvements_values.append(improvement)
pct_improvement = (improvement / gumroad_values[-1]) * 100 if gumroad_values[-1] > 0 else 0
improvement_percentages.append(pct_improvement)

summary_labels.append("Avg Relevant Rank")
gumroad_values.append(metrics['avg_relevant_rank']['gumroad'])
endpoint_values.append(metrics['avg_relevant_rank']['endpoint'])
improvement = gumroad_values[-1] - endpoint_values[-1]  # Note: reversed because lower is better
improvements_values.append(improvement)
pct_improvement = (improvement / gumroad_values[-1]) * 100 if gumroad_values[-1] > 0 else 0
improvement_percentages.append(pct_improvement)

# Create a table plot
table_data = []
for i in range(len(summary_labels)):
    if "Position" in summary_labels[i] or "Rank" in summary_labels[i]:
        # Format position metrics with fewer decimal places
        table_data.append([
            summary_labels[i],
            f"{gumroad_values[i]:.2f}",
            f"{endpoint_values[i]:.2f}",
            f"{improvements_values[i]:+.2f}",
            f"{improvement_percentages[i]:+.1f}%"
        ])
    else:
        # Format other metrics with 4 decimal places
        table_data.append([
            summary_labels[i],
            f"{gumroad_values[i]:.4f}",
            f"{endpoint_values[i]:.4f}",
            f"{improvements_values[i]:+.4f}",
            f"{improvement_percentages[i]:+.1f}%"
        ])

# Create the table
table = ax.table(
    cellText=table_data,
    colLabels=["Metric", "Gumroad", "Custom Endpoint", "Absolute Improvement", "Relative Improvement"],
    loc='center',
    cellLoc='center'
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)  # Adjust table size

# Color code the improvements
for i in range(len(summary_labels)):
    if improvement_percentages[i] > 0:
        table[(i+1, 4)].set_facecolor('#d8f3d8')  # Light green for positive improvements
    elif improvement_percentages[i] < 0:
        table[(i+1, 4)].set_facecolor('#f3d8d8')  # Light red for negative improvements

# Style the header
for j in range(5):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Remove axes
ax.axis('off')
ax.set_title('Search Engine Evaluation Metrics Summary', fontsize=18, pad=20)

plt.tight_layout()
plt.savefig('search_metrics_summary_table.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations generated successfully:")
print("1. search_metrics_dashboard.png - Dashboard of all metrics")
print("2. search_improvement_summary.png - Bar chart of relative improvements")
print("3. search_metrics_summary_table.png - Table summary of all metrics")
