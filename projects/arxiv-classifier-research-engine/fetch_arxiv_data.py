import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
import base64
from io import BytesIO

# === FETCH REAL ARXIV DATA ===
# Query arXiv API for recent papers in cs.LG, cs.AI, cs.CL

categories = ['cs.LG', 'cs.AI', 'cs.CL']
all_papers = []

print("Fetching real arXiv data...")

for cat in categories:
    query = urllib.parse.quote(f'cat:{cat}')
    url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=150&sortBy=submittedDate&sortOrder=descending'
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
    
    root = ET.fromstring(data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns)
        authors = entry.findall('atom:author', ns)
        summary = entry.find('atom:summary', ns)
        published = entry.find('atom:published', ns)
        categories_el = entry.findall('atom:category', ns)
        
        paper = {
            'title': title.text.strip() if title is not None else '',
            'authors': ', '.join([a.find('atom:name', ns).text for a in authors if a.find('atom:name', ns) is not None]),
            'abstract': summary.text.strip() if summary is not None else '',
            'published': published.text[:10] if published is not None else '',
            'primary_category': cat,
            'categories': [c.get('term', '') for c in categories_el]
        }
        all_papers.append(paper)
    print(f"  Fetched {len(root.findall('atom:entry', ns))} papers from {cat}")

print(f"\nTotal papers fetched: {len(all_papers)}")

# Save raw data
df = pd.DataFrame(all_papers)
df.to_csv('/tmp/genai-build/projects/arxiv-abstracts/data/arxiv_papers.csv', index=False)
with open('/tmp/genai-build/projects/arxiv-abstracts/data/arxiv_papers.json', 'w') as f:
    json.dump(all_papers, f, indent=2)

print("Data saved.")

# === ANALYSIS & VISUALIZATION ===

# 1. Category distribution
cat_counts = df['primary_category'].value_counts()

fig1, ax1 = plt.subplots(figsize=(10, 6))
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax1.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('arXiv Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
ax1.set_title('Recent Paper Distribution by Category\n(cs.LG, cs.AI, cs.CL)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(cat_counts.values) * 1.15)
for bar, val in zip(bars, cat_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
fig1.savefig('/tmp/genai-build/projects/arxiv-abstracts/figures/category_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig1)

# 2. Abstract length distribution by category
fig2, ax2 = plt.subplots(figsize=(12, 7))
df['abstract_length'] = df['abstract'].str.len()
for i, cat in enumerate(categories):
    subset = df[df['primary_category'] == cat]['abstract_length']
    ax2.hist(subset, bins=25, alpha=0.6, label=cat, color=colors[i], edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Abstract Length (characters)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Abstract Length Distribution by Category', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, framealpha=0.9)
ax2.axvline(df['abstract_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {df["abstract_length"].mean():.0f}')
plt.tight_layout()
fig2.savefig('/tmp/genai-build/projects/arxiv-abstracts/figures/abstract_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

# 3. Top keywords analysis
print("\nAnalyzing keywords...")
stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'using', 'based', 'proposed', 'approach', 'model', 'models', 'method', 'methods', 'results', 'paper', 'show', 'shown', 'demonstrate', 'we', 'our', 'us'}

all_words = []
for abstract in df['abstract']:
    words = abstract.lower().split()
    all_words.extend([w.strip('.,;:!?()[]{}"\'') for w in words if len(w) > 3 and w not in stopwords])

top_keywords = Counter(all_words).most_common(20)
keywords, counts = zip(*top_keywords)

fig3, ax3 = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(keywords))
ax3.barh(y_pos, counts, color='#2E86AB', edgecolor='black', linewidth=0.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(keywords, fontsize=11)
ax3.invert_yaxis()
ax3.set_xlabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Top 20 Keywords in arXiv Abstracts\n(cs.LG, cs.AI, cs.CL)', fontsize=14, fontweight='bold')
for i, v in enumerate(counts):
    ax3.text(v + 1, i, str(v), va='center', fontsize=10)
plt.tight_layout()
fig3.savefig('/tmp/genai-build/projects/arxiv-abstracts/figures/top_keywords.png', dpi=150, bbox_inches='tight')
plt.close(fig3)

# 4. Publication timeline
df['published_date'] = pd.to_datetime(df['published'])
df['month'] = df['published_date'].dt.to_period('M').astype(str)
monthly_counts = df.groupby('month').size()

fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(range(len(monthly_counts)), monthly_counts.values, marker='o', linewidth=2.5, markersize=8, color='#A23B72')
ax4.set_xticks(range(len(monthly_counts)))
ax4.set_xticklabels(monthly_counts.index, rotation=45, ha='right', fontsize=9)
ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
ax4.set_ylabel('Papers Published', fontsize=12, fontweight='bold')
ax4.set_title('Publication Timeline', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
for i, v in enumerate(monthly_counts.values):
    ax4.text(i, v + 1, str(v), ha='center', fontsize=9)
plt.tight_layout()
fig4.savefig('/tmp/genai-build/projects/arxiv-abstracts/figures/publication_timeline.png', dpi=150, bbox_inches='tight')
plt.close(fig4)

print("All 4 figures generated.")

# Save base64 encoded figures for notebook embedding
fig_data = {}
for fig_name in ['category_distribution', 'abstract_length_distribution', 'top_keywords', 'publication_timeline']:
    with open(f'/tmp/genai-build/projects/arxiv-abstracts/figures/{fig_name}.png', 'rb') as f:
        fig_data[fig_name] = base64.b64encode(f.read()).decode('utf-8')

with open('/tmp/genai-build/projects/arxiv-abstracts/data/figure_base64.json', 'w') as f:
    json.dump(fig_data, f, indent=2)

print("Base64 figures saved for notebook embedding.")
print(f"\nDataset summary: {len(df)} papers across {len(categories)} categories")
print(f"Date range: {df['published'].min()} to {df['published'].max()}")
