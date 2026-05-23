import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# LLM Document Classification — Quick Figure Generator
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('figures', exist_ok=True)

# Load the processed data
df = pd.read_parquet('data/processed/cleaned_documents.parquet')
labels = np.load('data/processed/y.npy', allow_pickle=True)

# Figure 1: Document length distribution
fig, ax = plt.subplots(figsize=(10, 6))
if 'text' in df.columns or 'content' in df.columns or 'document' in df.columns:
    text_col = 'text' if 'text' in df.columns else ('content' if 'content' in df.columns else 'document')
    lengths = df[text_col].str.len()
    ax.hist(lengths, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
    ax.axvline(lengths.median(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Median: {lengths.median():.0f} chars')
    ax.set_title('Document Length Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Character Count')
    ax.set_ylabel('Frequency')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'Document text column\nnot found', ha='center', va='center', fontsize=12)
    ax.set_title('Document Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/document_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Class distribution
fig, ax = plt.subplots(figsize=(10, 6))
unique, counts = np.unique(labels, return_counts=True)
# Decode labels if they're numeric
label_mapping = pd.read_csv('data/processed/label_mapping.csv') if os.path.exists('data/processed/label_mapping.csv') else None
if label_mapping is not None and 'label' in label_mapping.columns and 'encoded' in label_mapping.columns:
    mapping = dict(zip(label_mapping['encoded'], label_mapping['label']))
    label_names = [mapping.get(u, str(u)) for u in unique]
else:
    label_names = [str(u) for u in unique]

ax.bar(range(len(unique)), counts, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:len(unique)])
ax.set_xticks(range(len(unique)))
ax.set_xticklabels(label_names, rotation=45, ha='right')
ax.set_title('Document Class Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(counts):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Classification performance (from existing reports if available)
report_files = [f for f in os.listdir('reports') if f.endswith('.json')] if os.path.exists('reports') else []
if report_files:
    import json
    with open(f'reports/{report_files[0]}') as f:
        metrics = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_values = [metrics.get(m, 0) for m in metric_names]
    ax.bar([m.replace('_', ' ').title() for m in metric_names], metric_values, color=['#2ecc71', '#3498db', '#9b59b6', '#f39c12'])
    ax.set_ylim(0, 1)
    ax.set_title('Classification Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/classification_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"Generated {len(os.listdir('figures'))} figures for llm-document-classification")
