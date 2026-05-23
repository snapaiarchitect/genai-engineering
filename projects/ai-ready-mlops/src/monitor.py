#!/usr/bin/env python3
"""
monitor.py — Inference monitoring: latency, cost, accuracy tracking.

Logs every inference to SQLite time-series table.
Generates aggregated metrics: p50/p95/p99 latency, cost estimates,
rolling accuracy windows.

Usage:
    python src/monitor.py
    # Or import and call log_inference()
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.model_registry import ModelRegistry


BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)


class InferenceMonitor:
    """Monitor inference performance per model version."""

    def __init__(self, db_path: str = "data/metrics/inference_log.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            input_hash TEXT,
            prediction TEXT,
            ground_truth TEXT,
            latency_ms REAL NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_inf_exp ON inferences(experiment_name);
        CREATE INDEX IF NOT EXISTS idx_inf_time ON inferences(timestamp);
        CREATE INDEX IF NOT EXISTS idx_inf_run ON inferences(run_id);
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)

    def log_inference(
        self,
        experiment_name: str,
        run_id: int,
        model_version: str,
        latency_ms: float,
        prediction: Optional[str] = None,
        ground_truth: Optional[str] = None,
        input_hash: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO inferences
                   (experiment_name, run_id, model_version, input_hash, prediction, ground_truth, latency_ms, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (experiment_name, run_id, model_version, input_hash, prediction, ground_truth, latency_ms, now),
            )
            conn.commit()

    def get_latency_percentiles(
        self,
        experiment_name: str,
        window_hours: int = 24,
    ) -> Dict[str, float]:
        """Get p50, p95, p99 latency for an experiment."""
        since = (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT latency_ms FROM inferences WHERE experiment_name = ? AND timestamp > ?",
                (experiment_name, since),
            )
            latencies = [r[0] for r in cur.fetchall()]
        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0, "count": 0}
        arr = np.array(latencies)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "count": len(latencies),
            "mean": round(float(arr.mean()), 2),
        }

    def get_accuracy_window(
        self,
        experiment_name: str,
        window_hours: int = 24,
    ) -> Dict[str, Any]:
        """Rolling accuracy for inferences where ground_truth is available."""
        since = (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """SELECT prediction, ground_truth FROM inferences
                   WHERE experiment_name = ? AND timestamp > ? AND ground_truth IS NOT NULL""",
                (experiment_name, since),
            )
            rows = cur.fetchall()
        if not rows:
            return {"accuracy": None, "count": 0}
        correct = sum(1 for p, g in rows if p == g)
        total = len(rows)
        return {
            "accuracy": round(correct / total, 4) if total > 0 else None,
            "count": total,
        }

    def estimate_cost(self, experiment_name: str, cost_per_1k: float = 0.10) -> Dict[str, float]:
        """Estimate compute cost for last 24h of inferences."""
        since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM inferences WHERE experiment_name = ? AND timestamp > ?",
                (experiment_name, since),
            )
            count = cur.fetchone()[0]
        cost = (count / 1000) * cost_per_1k
        return {
            "inference_count": count,
            "cost_usd": round(cost, 4),
            "cost_per_1k": cost_per_1k,
        }

    def get_model_comparison(self, experiment_name: str) -> pd.DataFrame:
        """Compare latency/accuracy across model versions."""
        since = (datetime.utcnow() - timedelta(days=7)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT model_version, latency_ms, prediction, ground_truth
                   FROM inferences WHERE experiment_name = ? AND timestamp > ?""",
                (experiment_name, since),
            )
            df = pd.DataFrame([dict(r) for r in rows.fetchall()])
        if df.empty:
            return pd.DataFrame()
        summary = df.groupby("model_version").agg(
            count=("latency_ms", "count"),
            latency_p50=("latency_ms", lambda x: x.median()),
            latency_p95=("latency_ms", lambda x: x.quantile(0.95)),
        ).reset_index()
        return summary

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate all monitoring data for Streamlit dashboard."""
        data = {"experiments": {}, "alerts": [], "timestamp": datetime.utcnow().isoformat()}

        # Get all experiments from registry
        reg = ModelRegistry(str(DATA_DIR / "registry.db"))
        for exp in reg.list_experiments():
            exp_name = exp["name"]
            prod = reg.get_production_run(exp_name)
            data["experiments"][exp_name] = {
                "production_version": prod["version"] if prod else None,
                "production_run_id": prod["id"] if prod else None,
                "latency": self.get_latency_percentiles(exp_name, window_hours=24),
                "accuracy": self.get_accuracy_window(exp_name, window_hours=24),
                "cost": self.estimate_cost(exp_name),
            }
            # Alert if latency p95 > 200ms
            lat = data["experiments"][exp_name]["latency"]
            if lat["p95"] > 200:
                data["alerts"].append({
                    "experiment": exp_name,
                    "type": "latency",
                    "message": f"p95 latency {lat['p95']}ms exceeds 200ms threshold",
                    "severity": "warning" if lat["p95"] < 500 else "critical",
                })

        return data


def main():
    print("📊 Inference Monitor")
    monitor = InferenceMonitor()

    # Simulate some inference logs for demo
    experiments = [
        ("census-demographics", 1, "1.0.0"),
        ("bls-employment", 2, "1.0.0"),
        ("arxiv-abstracts", 3, "1.0.0"),
    ]
    for exp, rid, ver in experiments:
        for i in range(100):
            latency = max(1, np.random.normal(45, 12))
            monitor.log_inference(exp, rid, ver, latency_ms=round(latency, 2))

    # Generate report
    data = monitor.generate_dashboard_data()
    print(f"\nMonitoring data generated at {data['timestamp']}")
    for exp_name, metrics in data["experiments"].items():
        print(f"\n  {exp_name}:")
        print(f"    Production: {metrics['production_version']}")
        print(f"    Latency (p50/p95/p99): {metrics['latency']['p50']}/{metrics['latency']['p95']}/{metrics['latency']['p99']} ms")
        print(f"    24h cost: ${metrics['cost']['cost_usd']} ({metrics['cost']['inference_count']} inferences)")
    if data["alerts"]:
        print(f"\n  Alerts: {len(data['alerts'])}")
        for a in data["alerts"]:
            print(f"    [{a['severity'].upper()}] {a['message']}")
    else:
        print("\n  No alerts — all systems nominal.")

    # Save dashboard snapshot
    snapshot_path = METRICS_DIR / f"monitor_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(snapshot_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSnapshot saved: {snapshot_path}")


if __name__ == "__main__":
    main()
