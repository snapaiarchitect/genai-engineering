import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
os.makedirs(get_path('figures'), exist_ok=True)

df = pd.read_csv(get_path('data', 'congress_bills.csv'))
print(f'Total bills: {len(df)}')

# 1. Bill Type Distribution
type_counts = df['type'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('viridis', len(type_counts))
bars = ax.barh(type_counts.index, type_counts.values, color=colors)
for bar, val in zip(bars, type_counts.values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val} ({val/len(df)*100:.1f}%)', 
            va='center', fontsize=10)
ax.set_xlabel('Number of Bills')
ax.set_title('Bill Type Distribution (118th Congress)', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(type_counts.values) * 1.3)
plt.tight_layout()
plt.savefig(get_path('figures', 'bill_type_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'1. Bill types: {dict(type_counts)}')

# 2. Chamber Distribution
chamber_counts = df['origin_chamber'].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
colors = sns.color_palette('Set2', len(chamber_counts))
ax.pie(chamber_counts.values, labels=chamber_counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90, textprops={'fontsize': 12})
ax.set_title('Bills by Chamber of Origin', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'chamber_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'2. Chambers: {dict(chamber_counts)}')

# 3. Legislative Activity Timeline
df['latest_action_date'] = pd.to_datetime(df['latest_action_date'], errors='coerce')
monthly = df[df['latest_action_date'].notna()].set_index('latest_action_date').resample('M').size()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly.index, monthly.values, marker='o', color='steelblue', linewidth=2)
ax.fill_between(monthly.index, monthly.values, alpha=0.3, color='steelblue')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Actions')
ax.set_title('Legislative Activity Timeline', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'activity_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'3. Timeline: {len(monthly)} months of activity')

# 4. Bill Title Length Distribution
df['title_length'] = df['title'].str.len()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['title_length'], bins=50, color='coral', edgecolor='white', alpha=0.8)
ax.axvline(df['title_length'].median(), color='red', linestyle='--', label=f'Median: {df["title_length"].median():.0f}')
ax.axvline(df['title_length'].mean(), color='orange', linestyle='--', label=f'Mean: {df["title_length"].mean():.0f}')
ax.set_xlabel('Title Length (characters)')
ax.set_ylabel('Number of Bills')
ax.set_title('Bill Title Length Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(get_path('figures', 'title_length_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'4. Title length: median={df["title_length"].median():.0f}, mean={df["title_length"].mean():.0f}')

# 5. Top Keywords in Bill Titles
import re
all_words = []
for title in df['title'].dropna():
    words = re.findall(r'\b[A-Za-z]{4,}\b', title.lower())
    all_words.extend(words)

stop_words = {'bill', 'act', 'resolution', 'concurrent', 'joint', 'sense', 'congress', 'united', 'states', 'federal', 'national', 'shall', 'section', 'paragraph', 'subsection'}
filtered = [w for w in all_words if w not in stop_words]
word_counts = Counter(filtered).most_common(20)

fig, ax = plt.subplots(figsize=(10, 8))
words, counts = zip(*word_counts) if word_counts else ([], [])
colors = sns.color_palette('Spectral', len(words))
bars = ax.barh(range(len(words)), counts, color=colors)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=10)
ax.invert_yaxis()
for bar, val in zip(bars, counts):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
ax.set_xlabel('Frequency')
ax.set_title('Top 20 Keywords in Bill Titles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'title_keywords.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'5. Top keyword: {words[0] if words else "N/A"} ({counts[0] if counts else 0} occurrences)')

# 6. Action Type Analysis
df['action_type'] = df['latest_action_text'].str.extract(r'(Referred|Reported|Passed|Signed|Vetoed|Introduced|Discharged)')
action_counts = df['action_type'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('pastel', len(action_counts))
ax.bar(range(len(action_counts)), action_counts.values, color=colors)
ax.set_xticks(range(len(action_counts)))
ax.set_xticklabels(action_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Bills')
ax.set_title('Most Common Legislative Actions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'action_types.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'6. Action types: {dict(action_counts)}')

print(f'\nAll figures saved to {get_path("figures")}')
