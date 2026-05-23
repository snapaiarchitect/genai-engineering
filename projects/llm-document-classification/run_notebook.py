import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import os

os.chdir('/root/.openclaw/workspace/sierra-genai-engineering/projects/llm-document-classification')
os.makedirs('figures', exist_ok=True)

# Load data
df = pd.read_parquet('data/processed/cleaned_documents.parquet')
labels = np.load('data/processed/y.npy', allow_pickle=True)
X_tfidf = np.load('data/processed/X_tfidf.npy', allow_pickle=True)

print(f"Documents: {len(df)}")
print(f"Features: {X_tfidf.shape[1]}")
print(f"Classes: {len(np.unique(labels))}")

# Figure 1: Document length distribution
fig, ax = plt.subplots(figsize=(10, 6))
if 'text' in df.columns:
    lengths = df['text'].str.len()
    ax.hist(lengths, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
    ax.axvline(lengths.median(), color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Median: {lengths.median():.0f} chars')
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
label_mapping = pd.read_csv('data/processed/label_mapping.csv') if os.path.exists('data/processed/label_mapping.csv') else None
if label_mapping is not None and 'label' in label_mapping.columns and 'encoded' in label_mapping.columns:
    mapping = dict(zip(label_mapping['encoded'], label_mapping['label']))
    label_names = [mapping.get(u, str(u)) for u in unique]
else:
    label_names = [str(u) for u in unique]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
ax.bar(range(len(unique)), counts, color=colors[:len(unique)])
ax.set_xticks(range(len(unique)))
ax.set_xticklabels(label_names, rotation=45, ha='right')
ax.set_title('Document Class Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(counts):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Classification performance (train models)
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = (rf_pred == y_test).mean()

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = (lr_pred == y_test).mean()

print(f"RF Accuracy: {rf_acc:.3f}")
print(f"LR Accuracy: {lr_acc:.3f}")

# Classification report
from sklearn.metrics import precision_recall_fscore_support
rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='weighted')
lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='weighted')

fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
rf_values = [rf_acc, rf_prec, rf_rec, rf_f1]
lr_values = [lr_acc, lr_prec, lr_rec, lr_f1]

x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, rf_values, width, label='Random Forest', color='#3498db')
ax.bar(x + width/2, lr_values, width, label='Logistic Regression', color='#2ecc71')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_title('Classification Model Performance', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.legend()
for i, (v1, v2) in enumerate(zip(rf_values, lr_values)):
    ax.text(i - width/2, v1 + 0.02, f'{v1:.2f}', ha='center', fontweight='bold', fontsize=9)
    ax.text(i + width/2, v2 + 0.02, f'{v2:.2f}', ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('figures/classification_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Confusion matrix for Random Forest
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, rf_pred)
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=le.classes_, yticklabels=le.classes_,
       title='Random Forest Confusion Matrix',
       ylabel='True label', xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig('figures/confusion_matrix_rf.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: Feature importance (top 20)
fig, ax = plt.subplots(figsize=(10, 8))
feature_names = np.load('data/processed/feature_names.npy', allow_pickle=True) if os.path.exists('data/processed/feature_names.npy') else None
if feature_names is None:
    # Try to get from vectorizer
    import pickle
    if os.path.exists('data/processed/tfidf_vectorizer.pkl'):
        with open('data/processed/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        feature_names = np.array(vectorizer.get_feature_names_out())
    else:
        feature_names = np.array([f'feature_{i}' for i in range(X_tfidf.shape[1])])

importances = rf.feature_importances_
top_idx = np.argsort(importances)[-20:][::-1]
top_features = feature_names[top_idx]
top_scores = importances[top_idx]

ax.barh(range(len(top_features)), top_scores, color='#9b59b6')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features)
ax.invert_yaxis()
ax.set_title('Top 20 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Generated {len(os.listdir('figures'))} figures for LLM Document Classification")
