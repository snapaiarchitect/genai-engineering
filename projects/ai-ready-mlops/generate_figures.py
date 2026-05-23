#!/usr/bin/env python3
"""
generate_figures.py — Generate MLOps infrastructure visualization figures.

Run this after `python src/drift_detector.py` or `python src/model_registry.py`
to populate the SQLite registry with demo data.

Usage:
    python generate_figures.py

Outputs 4 PNG files to figures/:
    01_registry_versions.png    — Model version timeline (accuracy + latency)
    02_drift_detection.png      — PSI/KS drift detection dashboard
    03_ab_testing.png           — Traffic split + metric comparison
    04_monitoring_overview.png  — System health + request volume + latency percentiles + promotion pipeline
"""

import sys
sys.path.insert(0, 'src')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from src.model_registry import ModelRegistry
from src.drift_detector import DriftDetector
from src.ab_test import ABTestFramework

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
fig_dir = 'figures'

# ===== FIGURE 1: Model Registry Version Timeline =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Model versions over time
versions = ['v1.0', 'v1.1', 'v1.2', 'v2.0', 'v2.1']
accuracies = [0.82, 0.84, 0.85, 0.87, 0.88]
latencies = [45, 42, 40, 38, 35]
stages = ['staging', 'staging', 'production', 'staging', 'production']
colors = ['#FFA500' if s == 'staging' else '#00AA00' for s in stages]

ax1 = axes[0]
bars = ax1.bar(versions, accuracies, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Registry: Version Accuracy by Stage', fontsize=13, fontweight='bold')
ax1.set_ylim(0.78, 0.92)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Legend
staging_patch = mpatches.Patch(color='#FFA500', label='Staging')
prod_patch = mpatches.Patch(color='#00AA00', label='Production')
ax1.legend(handles=[staging_patch, prod_patch], loc='lower right')

# Right: Latency trend
ax2 = axes[1]
ax2.plot(versions, latencies, marker='o', markersize=8, linewidth=2, color='#1f77b4')
ax2.fill_between(versions, latencies, alpha=0.3, color='#1f77b4')
ax2.set_ylabel('Latency (ms)', fontsize=12)
ax2.set_title('Inference Latency Over Versions', fontsize=13, fontweight='bold')
ax2.set_ylim(30, 50)
for i, (v, l) in enumerate(zip(versions, latencies)):
    ax2.annotate(f'{l}ms', (i, l), textcoords="offset points", xytext=(0, 10), 
                 ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{fig_dir}/01_registry_versions.png', dpi=150, bbox_inches='tight')
print('Saved: 01_registry_versions.png')
plt.close()

# ===== FIGURE 2: Drift Detection Dashboard =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: PSI scores per feature
features = ['feature_A', 'feature_B', 'feature_C', 'feature_D', 'feature_E']
psi_scores = [0.03, 0.12, 0.28, 0.05, 0.15]
colors_psi = ['#00AA00' if p < 0.1 else '#FFA500' if p < 0.25 else '#FF0000' for p in psi_scores]

ax = axes[0, 0]
bars = ax.barh(features, psi_scores, color=colors_psi, edgecolor='black', linewidth=0.5)
ax.axvline(0.1, color='orange', linestyle='--', linewidth=1.5, label='Warning (0.1)')
ax.axvline(0.25, color='red', linestyle='--', linewidth=1.5, label='Critical (0.25)')
ax.set_xlabel('Population Stability Index (PSI)', fontsize=11)
ax.set_title('Drift Detection: PSI by Feature', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
for bar, psi in zip(bars, psi_scores):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{psi:.2f}', va='center', fontsize=10, fontweight='bold')

# Top-right: KS test p-values
p_values = [0.45, 0.08, 0.001, 0.32, 0.04]
colors_ks = ['#00AA00' if p > 0.05 else '#FF0000' for p in p_values]

ax = axes[0, 1]
bars = ax.barh(features, p_values, color=colors_ks, edgecolor='black', linewidth=0.5)
ax.axvline(0.05, color='red', linestyle='--', linewidth=1.5, label='alpha = 0.05')
ax.set_xlabel('Kolmogorov-Smirnov p-value', fontsize=11)
ax.set_title('Drift Detection: KS Test p-values', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
for bar, p in zip(bars, p_values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{p:.3f}', va='center', fontsize=10, fontweight='bold')

# Bottom-left: Feature distribution comparison (before vs after)
ax = axes[1, 0]
np.random.seed(42)
before = np.random.normal(0, 1, 1000)
after_drift = np.random.normal(0.3, 1.1, 1000)  # shifted mean, increased variance

ax.hist(before, bins=40, alpha=0.6, label='Training (Baseline)', color='#1f77b4', density=True)
ax.hist(after_drift, bins=40, alpha=0.6, label='Current Production', color='#ff7f0e', density=True)
ax.set_xlabel('Feature Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Feature Distribution: Baseline vs Production', fontsize=13, fontweight='bold')
ax.legend()

# Bottom-right: Drift alert timeline
ax = axes[1, 1]
dates = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
psi_timeline = [0.02, 0.03, 0.05, 0.08, 0.15, 0.28, 0.35]
status_colors = ['#00AA00', '#00AA00', '#00AA00', '#00AA00', '#FFA500', '#FF0000', '#FF0000']

ax.plot(dates, psi_timeline, marker='o', markersize=8, linewidth=2, color='#1f77b4')
ax.fill_between(dates, psi_timeline, alpha=0.3, color='#1f77b4')
ax.axhline(0.1, color='orange', linestyle='--', linewidth=1.5)
ax.axhline(0.25, color='red', linestyle='--', linewidth=1.5)
ax.set_ylabel('PSI', fontsize=11)
ax.set_title('7-Day Drift Alert Timeline', fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.4)

# Color-code points
for i, (d, p, c) in enumerate(zip(dates, psi_timeline, status_colors)):
    ax.scatter(d, p, color=c, s=100, zorder=5, edgecolor='black', linewidth=0.5)

# Status annotations (using text instead of emoji for font compatibility)
ax.text(6, 0.35, 'CRITICAL', fontsize=11, fontweight='bold', color='red', ha='right')
ax.text(4, 0.18, 'WARNING', fontsize=11, fontweight='bold', color='orange', ha='center')
ax.text(0, 0.05, 'STABLE', fontsize=11, fontweight='bold', color='green', ha='left')

plt.tight_layout()
plt.savefig(f'{fig_dir}/02_drift_detection.png', dpi=150, bbox_inches='tight')
print('Saved: 02_drift_detection.png')
plt.close()

# ===== FIGURE 3: A/B Test Router =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Traffic split over time
ax = axes[0]
hours = list(range(24))
control_traffic = [50] * 24
candidate_traffic = [50] * 24
# Gradual rollout: shift from 100/0 to 50/50
for i in range(12):
    control_traffic[i] = 100 - i * 4
    candidate_traffic[i] = i * 4

ax.fill_between(hours, control_traffic, alpha=0.6, label='Control (v1.2)', color='#1f77b4')
ax.fill_between(hours, candidate_traffic, alpha=0.6, label='Candidate (v2.0)', color='#ff7f0e')
ax.set_xlabel('Hour', fontsize=11)
ax.set_ylabel('Traffic %', fontsize=11)
ax.set_title('A/B Test: Gradual Traffic Rollout', fontsize=13, fontweight='bold')
ax.legend(loc='center right')
ax.set_ylim(0, 100)

# Right: Metric comparison
ax = axes[1]
metrics = ['Accuracy', 'F1 Score', 'Latency', 'Error Rate']
control_vals = [0.85, 0.83, 42, 0.05]
candidate_vals = [0.87, 0.86, 38, 0.03]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, control_vals, width, label='Control (v1.2)', color='#1f77b4', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, candidate_vals, width, label='Candidate (v2.0)', color='#ff7f0e', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Value', fontsize=11)
ax.set_title('A/B Test: Control vs Candidate Metrics', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Winner annotation
ax.annotate('WINNER', xy=(0.175, 0.87), xytext=(0.5, 0.92),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, fontweight='bold', color='green', ha='center')

plt.tight_layout()
plt.savefig(f'{fig_dir}/03_ab_testing.png', dpi=150, bbox_inches='tight')
print('Saved: 03_ab_testing.png')
plt.close()

# ===== FIGURE 4: Monitoring Dashboard Overview =====
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top: System health gauges (simulated)
ax1 = fig.add_subplot(gs[0, :])
health_metrics = ['Registry', 'Drift Detector', 'A/B Router', 'Retraining', 'Deployment']
health_scores = [99, 94, 97, 88, 100]
health_colors = ['#00AA00' if s > 95 else '#FFA500' if s > 80 else '#FF0000' for s in health_scores]

bars = ax1.barh(health_metrics, health_scores, color=health_colors, edgecolor='black', linewidth=0.5)
ax1.set_xlim(0, 110)
ax1.set_title('MLOps System Health Dashboard', fontsize=14, fontweight='bold', pad=10)
for bar, score in zip(bars, health_scores):
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f'{score}%', va='center', fontsize=11, fontweight='bold')

# Middle-left: Request volume
ax2 = fig.add_subplot(gs[1, 0])
hours = list(range(24))
volume = [120, 110, 95, 80, 75, 90, 150, 280, 420, 380, 350, 400,
          450, 430, 410, 390, 420, 480, 520, 450, 380, 300, 220, 160]
ax2.plot(hours, volume, color='#1f77b4', linewidth=2)
ax2.fill_between(hours, volume, alpha=0.3, color='#1f77b4')
ax2.set_title('Request Volume (24h)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Hour')

# Middle-center: Error rate
ax3 = fig.add_subplot(gs[1, 1])
error_rates = [0.02, 0.015, 0.01, 0.008, 0.012, 0.018, 0.025, 0.03, 0.028, 0.022, 0.02, 0.018,
               0.015, 0.012, 0.01, 0.008, 0.007, 0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02]
ax3.plot(hours, error_rates, color='#ff7f0e', linewidth=2)
ax3.axhline(0.02, color='red', linestyle='--', alpha=0.7, label='Alert threshold')
ax3.fill_between(hours, error_rates, alpha=0.3, color='#ff7f0e')
ax3.set_title('Error Rate (24h)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Hour')
ax3.legend()

# Middle-right: Latency percentiles
ax4 = fig.add_subplot(gs[1, 2])
p50 = [25, 24, 23, 22, 23, 26, 30, 35, 38, 36, 34, 32, 30, 28, 27, 26, 28, 32, 35, 33, 30, 28, 26, 25]
p95 = [45, 43, 42, 40, 42, 48, 55, 65, 72, 68, 62, 58, 55, 52, 50, 48, 52, 60, 68, 62, 55, 50, 48, 45]
p99 = [65, 62, 60, 58, 60, 68, 78, 90, 100, 95, 88, 82, 78, 75, 72, 70, 75, 85, 95, 88, 80, 75, 70, 65]

ax4.plot(hours, p50, label='p50', color='green', linewidth=1.5)
ax4.plot(hours, p95, label='p95', color='orange', linewidth=1.5)
ax4.plot(hours, p99, label='p99', color='red', linewidth=1.5)
ax4.fill_between(hours, p50, p95, alpha=0.2, color='orange')
ax4.fill_between(hours, p95, p99, alpha=0.2, color='red')
ax4.set_title('Latency Percentiles (ms)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Hour')
ax4.legend()

# Bottom: Model lineage
ax5 = fig.add_subplot(gs[2, :])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 3)
ax5.axis('off')
ax5.set_title('Model Promotion Pipeline', fontsize=13, fontweight='bold', pad=10)

# Draw pipeline stages
stages = [
    ('Training', 1, 1.5, '#6baed6'),
    ('Validation', 3, 1.5, '#9ecae1'),
    ('Staging', 5, 1.5, '#FFA500'),
    ('A/B Test', 7, 1.5, '#fd8d3c'),
    ('Production', 9, 1.5, '#00AA00'),
]
for name, x, y, color in stages:
    rect = mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=color, edgecolor='black', linewidth=1)
    ax5.add_patch(rect)
    ax5.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
for i in range(len(stages)-1):
    ax5.annotate('', xy=(stages[i+1][1]-0.4, stages[i+1][2]), 
                xytext=(stages[i][1]+0.4, stages[i][2]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Add rollback arrow
ax5.annotate('Rollback', xy=(5, 0.5), xytext=(9, 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
ax5.text(7, 0.3, 'Auto-rollback on drift', ha='center', fontsize=9, color='red', fontweight='bold')

plt.savefig(f'{fig_dir}/04_monitoring_overview.png', dpi=150, bbox_inches='tight')
print('Saved: 04_monitoring_overview.png')
plt.close()

print(f'\nAll 4 figures saved to {fig_dir}/')
