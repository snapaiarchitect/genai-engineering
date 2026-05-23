import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
os.makedirs(get_path('figures'), exist_ok=True)

df = pd.read_csv(get_path('data', 'clinical_trials.csv'))
print(f'Total trials: {len(df)}')

# 1. Phase Distribution
phase_counts = df['phase'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('viridis', len(phase_counts))
bars = ax.barh(phase_counts.index, phase_counts.values, color=colors)
for bar, val in zip(bars, phase_counts.values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val} ({val/len(df)*100:.1f}%)', 
            va='center', fontsize=10)
ax.set_xlabel('Number of Trials')
ax.set_title('Clinical Trial Phase Distribution (Active Cancer Trials)', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(phase_counts.values) * 1.2)
plt.tight_layout()
plt.savefig(get_path('figures', 'phase_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'1. Phase distribution: {dict(phase_counts)}')

# 2. Sponsor Distribution
sponsor_counts = df['sponsor_class'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
colors = sns.color_palette('Set2', len(sponsor_counts))
ax.pie(sponsor_counts.values, labels=sponsor_counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90, textprops={'fontsize': 11})
ax.set_title('Trial Sponsorship by Organization Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'sponsor_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'2. Sponsor distribution: {dict(sponsor_counts)}')

# 3. Enrollment Analysis
enrollment = pd.to_numeric(df['enrollment_count'], errors='coerce').dropna()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(enrollment, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Planned Enrollment')
axes[0].set_ylabel('Number of Trials')
axes[0].set_title('Enrollment Distribution')
axes[0].axvline(enrollment.median(), color='red', linestyle='--', label=f'Median: {enrollment.median():.0f}')
axes[0].axvline(enrollment.mean(), color='orange', linestyle='--', label=f'Mean: {enrollment.mean():.0f}')
axes[0].legend()

phase_df = df[df['phase'].notna()].copy()
phase_df['enrollment_count'] = pd.to_numeric(phase_df['enrollment_count'], errors='coerce')
phase_df = phase_df[phase_df['enrollment_count'] > 0]
sns.boxplot(data=phase_df, x='phase', y='enrollment_count', ax=axes[1], showfliers=False)
axes[1].set_xlabel('Trial Phase')
axes[1].set_ylabel('Planned Enrollment')
axes[1].set_title('Enrollment by Phase')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(get_path('figures', 'enrollment_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'3. Enrollment: median={enrollment.median():.0f}, mean={enrollment.mean():.0f}, max={enrollment.max():.0f}')

# 4. Top Conditions
df['primary_condition'] = df['condition'].str.split(',').str[0].str.strip()
top_conditions = df['primary_condition'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette('Spectral', len(top_conditions))
bars = ax.barh(range(len(top_conditions)), top_conditions.values, color=colors)
ax.set_yticks(range(len(top_conditions)))
ax.set_yticklabels(top_conditions.index, fontsize=9)
ax.invert_yaxis()
for bar, val in zip(bars, top_conditions.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
ax.set_xlabel('Number of Trials')
ax.set_title('Top 15 Cancer Types by Trial Volume', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'condition_frequency.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'4. Top condition: {top_conditions.index[0]} ({top_conditions.iloc[0]} trials)')

# 5. Study Type & Intervention
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
study_type = df['study_type'].value_counts()
axes[0].pie(study_type.values, labels=study_type.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Study Type Distribution')
intervention = df['intervention_type'].value_counts().head(8)
axes[1].bar(range(len(intervention)), intervention.values, color=sns.color_palette('pastel', len(intervention)))
axes[1].set_xticks(range(len(intervention)))
axes[1].set_xticklabels(intervention.index, rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Number of Trials')
axes[1].set_title('Top Intervention Types')
plt.tight_layout()
plt.savefig(get_path('figures', 'study_intervention.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'5. Study types: {dict(study_type)}')

# 6. Geographic Spread
location_counts = df['locations'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(location_counts.index, location_counts.values, color='coral', edgecolor='white')
ax.set_xlabel('Number of Study Locations')
ax.set_ylabel('Number of Trials')
ax.set_title('Geographic Distribution: How Many Sites per Trial?')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(get_path('figures', 'geographic_spread.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'6. Single-site: {(df["locations"] == 1).sum()}, Multi-site: {(df["locations"] > 1).sum()}, Max sites: {df["locations"].max()}')

print(f'\nAll figures saved to ../figures/')
print(f'Fetch date: {datetime.now().isoformat()}')
