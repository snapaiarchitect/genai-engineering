import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

os.makedirs(get_path('figures'), exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv(get_path('data', 'scotus_cases.csv'))
print(f'Total cases: {len(df)}')

# Extract justice vote columns
vote_cols = [c for c in df.columns if c.startswith('vote_')]
opinion_cols = [c for c in df.columns if c.startswith('opinion_')]
justices = [c.replace('vote_', '').replace('_', ' ') for c in vote_cols]
print(f'Justices: {justices}')

# 1. Vote Margin Distribution
margins = df['majority_votes'] - df['minority_votes']
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(margins, bins=range(0, 10), color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(margins.median(), color='red', linestyle='--', label=f'Median: {margins.median():.1f}')
ax.set_xlabel('Vote Margin (Majority - Minority)')
ax.set_ylabel('Number of Cases')
ax.set_title('SCOTUS Vote Margin Distribution (2022 Term)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(get_path('figures', 'vote_margin_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'1. Vote margins: median={margins.median():.1f}, mean={margins.mean():.1f}')

# 2. Justice Voting Alignment (co-occurrence matrix)
alignment = pd.DataFrame(0, index=justices, columns=justices)
for _, case in df.iterrows():
    for j1 in justices:
        for j2 in justices:
            if j1 == j2:
                continue
            v1 = case.get(f'vote_{j1.replace(" ", "_")}', '')
            v2 = case.get(f'vote_{j2.replace(" ", "_")}', '')
            if v1 and v2 and v1 == v2:
                alignment.loc[j1, j2] += 1

# Normalize by total cases both participated in
for j1 in justices:
    for j2 in justices:
        if j1 != j2:
            both_present = df[[f'vote_{j1.replace(" ", "_")}', f'vote_{j2.replace(" ", "_")}']].notna().all(axis=1).sum()
            if both_present > 0:
                alignment.loc[j1, j2] = alignment.loc[j1, j2] / both_present

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(alignment, dtype=bool))
sns.heatmap(alignment, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
            vmin=0, vmax=1, square=True, ax=ax, cbar_kws={'label': 'Agreement Rate'})
ax.set_title('Justice Voting Alignment Matrix (2022 Term)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'justice_alignment_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'2. Justice alignment matrix saved')

# 3. Opinion Type Distribution
all_opinions = []
for col in opinion_cols:
    opinions = df[col].dropna().tolist()
    all_opinions.extend(opinions)

opinion_counts = Counter(all_opinions)
fig, ax = plt.subplots(figsize=(8, 6))
colors = sns.color_palette('Set2', len(opinion_counts))
ax.pie(opinion_counts.values(), labels=opinion_counts.keys(), autopct='%1.1f%%',
       colors=colors, startangle=90)
ax.set_title('Opinion Type Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'opinion_type_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'3. Opinion types: {dict(opinion_counts)}')

# 4. Case Timeline Analysis
df['date_argued'] = pd.to_datetime(df['date_argued'], errors='coerce')
df['date_decided'] = pd.to_datetime(df['date_decided'], errors='coerce')
df['decision_days'] = (df['date_decided'] - df['date_argued']).dt.days

timeline = df[df['decision_days'].notna() & (df['decision_days'] > 0)]
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(timeline['decision_days'], bins=30, color='coral', edgecolor='white', alpha=0.8)
ax.axvline(timeline['decision_days'].median(), color='red', linestyle='--', 
           label=f'Median: {timeline["decision_days"].median():.0f} days')
ax.axvline(timeline['decision_days'].mean(), color='orange', linestyle='--', 
           label=f'Mean: {timeline["decision_days"].mean():.0f} days')
ax.set_xlabel('Days from Argument to Decision')
ax.set_ylabel('Number of Cases')
ax.set_title('Case Timeline: Argument to Decision', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(get_path('figures', 'case_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'4. Timeline: median={timeline["decision_days"].median():.0f} days, mean={timeline["decision_days"].mean():.0f} days')

# 5. Majority Vote Distribution
fig, ax = plt.subplots(figsize=(8, 6))
vote_counts = df['majority_votes'].value_counts().sort_index()
colors = sns.color_palette('viridis', len(vote_counts))
ax.bar(vote_counts.index, vote_counts.values, color=colors, edgecolor='white')
for x, y in zip(vote_counts.index, vote_counts.values):
    ax.text(x, y + 0.3, str(y), ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Majority Votes')
ax.set_ylabel('Number of Cases')
ax.set_title('Majority Vote Count Distribution', fontsize=14, fontweight='bold')
ax.set_xticks(range(min(vote_counts.index), max(vote_counts.index) + 1))
plt.tight_layout()
plt.savefig(get_path('figures', 'majority_vote_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'5. Majority votes: {dict(vote_counts)}')

# 6. Case Topic Analysis from Questions
all_words = []
for q in df['question'].dropna():
    words = re.findall(r'\b[A-Za-z]{5,}\b', q.lower())
    all_words.extend(words)

stop_words = {'whether', 'court', 'federal', 'state', 'section', 'states', 'united', 'petition', 'petitioner', 'respondent', 'appellant', 'appellee', 'plaintiff', 'defendant'}
filtered = [w for w in all_words if w not in stop_words]
topic_counts = Counter(filtered).most_common(20)

fig, ax = plt.subplots(figsize=(10, 8))
if topic_counts:
    words, counts = zip(*topic_counts)
    colors = sns.color_palette('Spectral', len(words))
    ax.barh(range(len(words)), counts, color=colors)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    for bar, val in zip(ax.patches, counts):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
ax.set_xlabel('Frequency')
ax.set_title('Top Keywords in Case Questions Presented', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'case_topic_keywords.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'6. Top keyword: {topic_counts[0] if topic_counts else "N/A"}')

print(f'\nAll figures saved to {get_path("figures")}')
