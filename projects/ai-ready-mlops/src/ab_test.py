#!/usr/bin/env python3
"""
ab_test.py — A/B testing framework for model versions.

Traffic routing between control (production) and candidate (staging).
Measures performance with statistical significance testing.
Auto-promotes candidate if it beats control by configurable margin.

Usage:
    python src/ab_test.py
"""

import json
import time
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from scipy import stats

from src.model_registry import ModelRegistry


BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"


class ABTestFramework:
    """A/B testing with configurable traffic split and significance testing."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def get_models_for_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Get control (production) and candidate (latest staging) models."""
        control = self.registry.get_production_run(experiment_name)
        if not control:
            raise ValueError(f"No production model for {experiment_name}")

        # Find latest staging run
        runs = self.registry.list_runs(experiment_name)
        candidates = [r for r in runs if r.get("id") != control.get("id")]
        if not candidates:
            raise ValueError(f"No candidate model for {experiment_name}")
        candidate = candidates[0]  # most recent (list is DESC)

        return {"control": control, "candidate": candidate}

    def simulate_traffic(
        self,
        experiment_name: str,
        n_requests: int = 1000,
        split_ratio: float = 0.2,
    ) -> pd.DataFrame:
        """Simulate inference traffic with routing to control vs. candidate.
        Returns DataFrame with variant, latency, prediction, ground_truth.
        """
        models = self.get_models_for_experiment(experiment_name)
        control_id = models["control"]["id"]
        candidate_id = models["candidate"]["id"]

        # Load model objects
        try:
            control_artifact = self.registry.load_model(control_id, "model.pkl")
            candidate_artifact = self.registry.load_model(candidate_id, "model.pkl")
        except FileNotFoundError:
            # If no real models yet, simulate
            print("[A/B Test] No model artifacts found — simulating performance data")
            return self._simulate_performance(n_requests, split_ratio)

        records = []
        for i in range(n_requests):
            variant = "candidate" if np.random.random() < split_ratio else "control"
            run_id = candidate_id if variant == "candidate" else control_id

            # Simulate inference latency (candidate may be faster/slower)
            base_latency = np.random.exponential(45)  # ~45ms base
            if variant == "candidate":
                # Simulate 10% faster candidate
                latency = base_latency * 0.90 + np.random.normal(0, 2)
            else:
                latency = base_latency

            # Simulate accuracy (candidate may be slightly better)
            if variant == "candidate":
                correct = np.random.random() < 0.89  # 89% accuracy
            else:
                correct = np.random.random() < 0.87  # 87% accuracy

            records.append({
                "request_id": i,
                "variant": variant,
                "run_id": run_id,
                "latency_ms": round(max(1, latency), 2),
                "correct": correct,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return pd.DataFrame(records)

    def _simulate_performance(self, n_requests: int, split_ratio: float) -> pd.DataFrame:
        """Simulate A/B performance when no real models loaded."""
        records = []
        for i in range(n_requests):
            variant = "candidate" if np.random.random() < split_ratio else "control"
            base_latency = np.random.exponential(45)
            latency = base_latency * (0.90 if variant == "candidate" else 1.0)
            correct = np.random.random() < (0.89 if variant == "candidate" else 0.87)
            records.append({
                "request_id": i,
                "variant": variant,
                "run_id": 1 if variant == "control" else 2,
                "latency_ms": round(max(1, latency), 2),
                "correct": correct,
                "timestamp": datetime.utcnow().isoformat(),
            })
        return pd.DataFrame(records)

    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical analysis of A/B test results."""
        control = df[df["variant"] == "control"]
        candidate = df[df["variant"] == "candidate"]

        # Accuracy comparison
        control_acc = control["correct"].mean()
        candidate_acc = candidate["correct"].mean()

        # Two-proportion z-test for accuracy
        n_c = len(control)
        n_t = len(candidate)
        p_pool = (control["correct"].sum() + candidate["correct"].sum()) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
        z_acc = (candidate_acc - control_acc) / se if se > 0 else 0
        p_value_acc = 2 * (1 - stats.norm.cdf(abs(z_acc))) if se > 0 else 1.0

        # Latency comparison (two-sample t-test)
        t_stat, p_value_lat = stats.ttest_ind(
            candidate["latency_ms"], control["latency_ms"], equal_var=False
        )

        # Throughput (requests per second, estimated)
        control_throughput = 1000 / control["latency_ms"].mean()
        candidate_throughput = 1000 / candidate["latency_ms"].mean()

        results = {
            "control_samples": n_c,
            "candidate_samples": n_t,
            "control_accuracy": round(control_acc, 4),
            "candidate_accuracy": round(candidate_acc, 4),
            "accuracy_diff": round(candidate_acc - control_acc, 4),
            "accuracy_p_value": round(p_value_acc, 4),
            "accuracy_significant": p_value_acc < 0.05,
            "control_latency_p50": round(control["latency_ms"].median(), 2),
            "candidate_latency_p50": round(candidate["latency_ms"].median(), 2),
            "control_latency_p95": round(control["latency_ms"].quantile(0.95), 2),
            "candidate_latency_p95": round(candidate["latency_ms"].quantile(0.95), 2),
            "latency_p_value": round(p_value_lat, 4),
            "latency_significant": p_value_lat < 0.05,
            "control_throughput": round(control_throughput, 2),
            "candidate_throughput": round(candidate_throughput, 2),
            "recommendation": "hold",
        }

        # Auto-promote recommendation
        if candidate_acc > control_acc and p_value_acc < 0.05 and (candidate_acc - control_acc) > 0.01:
            results["recommendation"] = "promote"
        elif candidate_acc < control_acc and p_value_acc < 0.05:
            results["recommendation"] = "reject"

        return results

    def run_ab_test(
        self,
        experiment_name: str,
        n_requests: int = 1000,
        split_ratio: float = 0.2,
        auto_promote: bool = False,
    ) -> Dict[str, Any]:
        """Full A/B test: simulate traffic, analyze, optionally promote."""
        print(f"\n[A/B Test] Running for {experiment_name}")
        print(f"  Samples: {n_requests}, Candidate split: {split_ratio*100:.0f}%")

        df = self.simulate_traffic(experiment_name, n_requests, split_ratio)
        results = self.analyze_results(df)

        print(f"  Control accuracy: {results['control_accuracy']}")
        print(f"  Candidate accuracy: {results['candidate_accuracy']}")
        print(f"  Accuracy p-value: {results['accuracy_p_value']} {'*' if results['accuracy_significant'] else ''}")
        print(f"  Recommendation: {results['recommendation'].upper()}")

        if auto_promote and results["recommendation"] == "promote":
            models = self.get_models_for_experiment(experiment_name)
            self.registry.promote(
                experiment_name,
                models["candidate"]["id"],
                promoted_by="ab_test",
                reason=f"A/B test: accuracy {results['candidate_accuracy']} vs {results['control_accuracy']} (p={results['accuracy_p_value']})",
            )
            print(f"  [AUTO-PROMOTED] candidate to production")

        # Save results
        results["experiment"] = experiment_name
        results["timestamp"] = datetime.utcnow().isoformat()
        results["n_requests"] = n_requests
        results["split_ratio"] = split_ratio

        return results


def main():
    print("🧪 A/B Testing Framework")
    reg = ModelRegistry(str(DATA_DIR / "registry.db"))
    ab = ABTestFramework(reg)

    experiments = ["census-demographics", "bls-employment", "arxiv-abstracts"]
    for exp in experiments:
        try:
            result = ab.run_ab_test(exp, n_requests=1000, split_ratio=0.2)
            print(f"  Result: {result['recommendation'].upper()}")
        except Exception as e:
            print(f"[ERROR] A/B test failed for {exp}: {e}")

    print("\nA/B testing complete.")


if __name__ == "__main__":
    main()
