import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

os.makedirs(get_path('figures'), exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv(get_path('data', 'pubmed_abstracts.csv'))
print(f'Total articles: {len(df)}')

# 1. Publication Year Distribution
year_counts = df['year'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(year_counts.index.astype(str), year_counts.values, color='steelblue', edgecolor='white')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Articles')
ax.set_title('PubMed Article Publication Timeline (Machine Learning)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(get_path('figures', 'publication_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'1. Timeline: {len(year_counts)} years, peak={year_counts.max()} in {year_counts.idxmax()}')

# 2. Top Journals
journal_counts = df['journal'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette('viridis', len(journal_counts))
ax.barh(journal_counts.index, journal_counts.values, color=colors)
for bar, val in zip(ax.patches, journal_counts.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
ax.set_xlabel('Number of Articles')
ax.set_title('Top 15 Journals', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'journal_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'2. Top journal: {journal_counts.index[0]} ({journal_counts.iloc[0]} articles)')

# 3. Abstract Length Distribution
df['abstract_length'] = df['abstract'].str.len()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['abstract_length'], bins=50, color='coral', edgecolor='white', alpha=0.8)
ax.axvline(df['abstract_length'].median(), color='red', linestyle='--', label=f'Median: {df["abstract_length"].median():.0f}')
ax.axvline(df['abstract_length'].mean(), color='orange', linestyle='--', label=f'Mean: {df["abstract_length"].mean():.0f}')
ax.set_xlabel('Abstract Length (characters)')
ax.set_ylabel('Number of Articles')
ax.set_title('Abstract Length Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(get_path('figures', 'abstract_length_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'3. Abstract length: median={df["abstract_length"].median():.0f}, mean={df["abstract_length"].mean():.0f}')

# 4. Author Count Distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['author_count'], bins=range(0, df['author_count'].max()+2), color='seagreen', edgecolor='white')
ax.axvline(df['author_count'].median(), color='red', linestyle='--', label=f'Median: {df["author_count"].median():.0f}')
ax.set_xlabel('Number of Authors')
ax.set_ylabel('Number of Articles')
ax.set_title('Author Count Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(get_path('figures', 'author_count_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'4. Author count: median={df["author_count"].median():.0f}, mean={df["author_count"].mean():.1f}')

# 5. Top MeSH Terms
all_mesh = []
for terms in df['mesh_terms'].dropna():
    all_mesh.extend([t.strip() for t in terms.split(';') if t.strip()])

mesh_counts = Counter(all_mesh).most_common(20)
fig, ax = plt.subplots(figsize=(10, 8))
if mesh_counts:
    terms, counts = zip(*mesh_counts)
    colors = sns.color_palette('Spectral', len(terms))
    ax.barh(range(len(terms)), counts, color=colors)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=9)
    ax.invert_yaxis()
    for bar, val in zip(ax.patches, counts):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=8)
ax.set_xlabel('Frequency')
ax.set_title('Top 20 MeSH Terms', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'mesh_terms.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'5. Top MeSH: {mesh_counts[0] if mesh_counts else "N/A"}')

# 6. Title Keyword Analysis
import re
all_words = []
for title in df['title'].dropna():
    words = re.findall(r'\b[A-Za-z]{5,}\b', title.lower())
    all_words.extend(words)

stop_words = {'using', 'based', 'analysis', 'model', 'models', 'study', 'effects', 'effect', 'patients', 'patient', 'clinical', 'treatment', 'method', 'methods', 'results', 'review', 'systematic', 'randomized', 'controlled', 'trial', 'data', 'learning', 'machine'}
filtered = [w for w in all_words if w not in stop_words]
word_counts = Counter(filtered).most_common(20)

fig, ax = plt.subplots(figsize=(10, 8))
if word_counts:
    words, counts = zip(*word_counts)
    colors = sns.color_palette('husl', len(words))
    ax.barh(range(len(words)), counts, color=colors)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    for bar, val in zip(ax.patches, counts):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
ax.set_xlabel('Frequency')
ax.set_title('Top Keywords in Article Titles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(get_path('figures', 'title_keywords.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'6. Top title keyword: {word_counts[0] if word_counts else "N/A"}')

print(f'\nAll figures saved to {get_path("figures")}')
