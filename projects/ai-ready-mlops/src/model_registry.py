#!/usr/bin/env python3
"""
model_registry.py — MLflow-style experiment tracking with SQLite backend.

Zero external MLflow server needed. This module provides:
- Experiment and run CRUD
- Metric and parameter logging
- Artifact storage (models, metrics, params as JSON/pickle)
- Tag-based model promotion (staging → production)
- Query interface: best run, compare runs, get production model

Usage:
    from src.model_registry import ModelRegistry
    reg = ModelRegistry("data/registry.db")
    run_id = reg.start_run("census-demographics", {"n_estimators": 100})
    reg.log_metric(run_id, "accuracy", 0.87)
    reg.save_model(run_id, model_obj, "census_classifier.pkl")
    reg.set_tag(run_id, "stage", "production")
"""

import sqlite3
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """SQLite-backed model registry with MLflow-style API."""

    def __init__(self, db_path: str = "data/registry.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.db_path.parent / "models"
        self.artifact_dir.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create registry schema if not exists."""
        schema = """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            run_name TEXT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            params_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            tag_key TEXT NOT NULL,
            tag_value TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );
        CREATE TABLE IF NOT EXISTS promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            from_run_id INTEGER,
            to_run_id INTEGER NOT NULL,
            promoted_by TEXT,
            reason TEXT,
            metrics_at_promotion TEXT,
            promoted_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
        CREATE INDEX IF NOT EXISTS idx_tags_run ON tags(run_id);
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)

    # ── Experiment Management ────────────────────────────────────────────

    def get_or_create_experiment(self, name: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT id FROM experiments WHERE name = ?", (name,))
            row = cur.fetchone()
            if row:
                return row[0]
            now = datetime.utcnow().isoformat()
            cur = conn.execute(
                "INSERT INTO experiments (name, created_at) VALUES (?, ?)",
                (name, now)
            )
            conn.commit()
            return cur.lastrowid

    def list_experiments(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC")
            return [dict(r) for r in rows.fetchall()]

    # ── Run Management ─────────────────────────────────────────────────────

    def start_run(
        self,
        experiment_name: str,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
    ) -> int:
        """Start a new run. Returns run_id. Auto-increments version."""
        exp_id = self.get_or_create_experiment(experiment_name)
        version = self._next_version(experiment_name, model_name)
        now = datetime.utcnow().isoformat()
        params_json = json.dumps(params or {})
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO runs
                   (experiment_id, run_name, model_name, version, params_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (exp_id, run_name or f"{model_name}-{version}", model_name, version, params_json, now),
            )
            conn.commit()
            return cur.lastrowid

    def _next_version(self, experiment_name: str, model_name: str) -> str:
        """Auto-increment patch version: 1.0.0 → 1.0.1."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """SELECT MAX(version) FROM runs
                   WHERE experiment_id = (SELECT id FROM experiments WHERE name = ?)
                   AND model_name = ?""",
                (experiment_name, model_name),
            )
            row = cur.fetchone()
            latest = row[0] if row and row[0] else "0.0.0"
        parts = [int(p) for p in latest.split(".")]
        parts[2] += 1  # patch increment
        return ".".join(str(p) for p in parts)

    def log_metric(self, run_id: int, metric_name: str, value: float) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, metric_name, value, timestamp) VALUES (?, ?, ?, ?)",
                (run_id, metric_name, value, now),
            )
            conn.commit()

    def log_metrics(self, run_id: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(run_id, k, v)

    def set_tag(self, run_id: int, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tags (run_id, tag_key, tag_value) VALUES (?, ?, ?)",
                (run_id, key, value),
            )
            conn.commit()

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            run = dict(row)
            run["params"] = json.loads(run.pop("params_json") or "{}")
            # fetch metrics
            mcur = conn.execute("SELECT metric_name, value FROM metrics WHERE run_id = ?", (run_id,))
            run["metrics"] = {r["metric_name"]: r["value"] for r in mcur.fetchall()}
            # fetch tags
            tcur = conn.execute("SELECT tag_key, tag_value FROM tags WHERE run_id = ?", (run_id,))
            run["tags"] = {r["tag_key"]: r["tag_value"] for r in tcur.fetchall()}
            return run

    def list_runs(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if experiment_name:
                rows = conn.execute(
                    """SELECT r.* FROM runs r
                       JOIN experiments e ON r.experiment_id = e.id
                       WHERE e.name = ? ORDER BY r.created_at DESC""",
                    (experiment_name,),
                )
            else:
                rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC")
            results = []
            for r in rows.fetchall():
                run = dict(r)
                run["params"] = json.loads(run.pop("params_json") or "{}")
                results.append(run)
            return results

    def get_best_run(self, experiment_name: str, metric: str = "accuracy", mode: str = "max") -> Optional[Dict[str, Any]]:
        """Get the run with best metric value."""
        order = "DESC" if mode == "max" else "ASC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                f"""SELECT r.*, m.value as best_metric FROM runs r
                    JOIN experiments e ON r.experiment_id = e.id
                    JOIN metrics m ON r.id = m.run_id
                    WHERE e.name = ? AND m.metric_name = ?
                    ORDER BY m.value {order} LIMIT 1""",
                (experiment_name, metric),
            )
            row = cur.fetchone()
            if not row:
                return None
            run = dict(row)
            run["params"] = json.loads(run.pop("params_json") or "{}")
            run["metrics"] = {metric: run.pop("best_metric")}
            return run

    # ── Artifact Management ──────────────────────────────────────────────

    def _artifact_path(self, run_id: int, filename: str) -> Path:
        exp_name = self._get_experiment_name_for_run(run_id)
        path = self.artifact_dir / exp_name / str(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def _get_experiment_name_for_run(self, run_id: int) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """SELECT e.name FROM experiments e
                   JOIN runs r ON r.experiment_id = e.id WHERE r.id = ?""",
                (run_id,),
            )
            row = cur.fetchone()
            return row[0] if row else "unknown"

    def save_model(self, run_id: int, model: Any, filename: str = "model.pkl") -> str:
        """Serialize model with joblib. Returns artifact path."""
        path = self._artifact_path(run_id, filename)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return str(path)

    def load_model(self, run_id: int, filename: str = "model.pkl") -> Any:
        path = self._artifact_path(run_id, filename)
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_json(self, run_id: int, data: Dict[str, Any], filename: str) -> str:
        path = self._artifact_path(run_id, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)

    def load_json(self, run_id: int, filename: str) -> Dict[str, Any]:
        path = self._artifact_path(run_id, filename)
        with open(path, "r") as f:
            return json.load(f)

    # ── Deployment & Promotion ───────────────────────────────────────────

    def get_production_run(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get the run currently tagged as 'production' for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """SELECT r.* FROM runs r
                   JOIN experiments e ON r.experiment_id = e.id
                   JOIN tags t ON r.id = t.run_id
                   WHERE e.name = ? AND t.tag_key = 'stage' AND t.tag_value = 'production'
                   ORDER BY r.created_at DESC LIMIT 1""",
                (experiment_name,),
            )
            row = cur.fetchone()
            if not row:
                return None
            run = dict(row)
            run["params"] = json.loads(run.pop("params_json") or "{}")
            return run

    def promote(
        self,
        experiment_name: str,
        to_run_id: int,
        promoted_by: str = "system",
        reason: str = "",
    ) -> None:
        """Promote a run to production. Demotes previous production run."""
        now = datetime.utcnow().isoformat()
        # Get current production run
        prev = self.get_production_run(experiment_name)
        prev_id = prev["id"] if prev else None

        # Demote previous
        if prev_id:
            self.set_tag(prev_id, "stage", "archived")

        # Promote new
        self.set_tag(to_run_id, "stage", "production")

        # Log promotion
        run = self.get_run(to_run_id)
        metrics_json = json.dumps(run.get("metrics", {})) if run else "{}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO promotions
                   (experiment_name, from_run_id, to_run_id, promoted_by, reason, metrics_at_promotion, promoted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (experiment_name, prev_id, to_run_id, promoted_by, reason, metrics_json, now),
            )
            conn.commit()

    def rollback(self, experiment_name: str, to_run_id: int, reason: str = "") -> None:
        """Rollback production to a previous run."""
        self.promote(experiment_name, to_run_id, promoted_by="rollback", reason=reason)

    def get_promotion_history(self, experiment_name: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM promotions WHERE experiment_name = ? ORDER BY promoted_at DESC",
                (experiment_name,),
            )
            return [dict(r) for r in rows.fetchall()]


# ── CLI Demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🗄️  Model Registry Demo")
    reg = ModelRegistry("data/registry.db")

    # Demo: census experiment
    exp = "census-demographics"
    rid = reg.start_run(exp, "census_classifier", {"n_estimators": 100, "max_depth": 10})
    print(f"Started run {rid} for {exp}")
    reg.log_metrics(rid, {"accuracy": 0.87, "f1": 0.84, "latency_ms": 45.2})
    reg.set_tag(rid, "stage", "production")
    print(f"Run details: {reg.get_run(rid)}")

    # Best run query
    best = reg.get_best_run(exp, metric="accuracy")
    print(f"Best run: {best['version']} with accuracy={best['metrics'].get('accuracy')}")

    # List experiments
    print(f"Experiments: {[e['name'] for e in reg.list_experiments()]}")
    print("Registry demo complete.")
