#!/usr/bin/env python3
"""
deploy.py — Model promotion workflow: staging → production with rollback.

Usage:
    python src/deploy.py --promote census-demographics --to-run 5 --reason "A/B test passed"
    python src/deploy.py --rollback census-demographics --to-version 1.0.0 --reason "latency regression"
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.model_registry import ModelRegistry


BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"


def promote(experiment_name: str, to_run_id: int, reason: str, by: str = "user"):
    """Promote a run to production."""
    reg = ModelRegistry(str(DATA_DIR / "registry.db"))
    run = reg.get_run(to_run_id)
    if not run:
        print(f"❌ Run {to_run_id} not found")
        return False
    if run.get("tags", {}).get("stage") == "production":
        print(f"⚠️  Run {to_run_id} is already production")
        return True

    reg.promote(experiment_name, to_run_id, promoted_by=by, reason=reason)
    print(f"✅ Promoted {experiment_name} run {to_run_id} (v{run['version']}) to production")
    print(f"   Reason: {reason}")
    return True


def rollback(experiment_name: str, to_version: str, reason: str):
    """Rollback to a previous version."""
    reg = ModelRegistry(str(DATA_DIR / "registry.db"))
    runs = reg.list_runs(experiment_name)
    target = [r for r in runs if r["version"] == to_version]
    if not target:
        print(f"❌ Version {to_version} not found for {experiment_name}")
        return False
    to_run_id = target[0]["id"]
    reg.rollback(experiment_name, to_run_id, reason=reason)
    print(f"⏪ Rolled back {experiment_name} to v{to_version} (run {to_run_id})")
    print(f"   Reason: {reason}")

    # Show promotion history
    history = reg.get_promotion_history(experiment_name)
    print(f"\n   Promotion history ({len(history)} events):")
    for h in history[:5]:
        print(f"     {h['promoted_at']}: v{h['to_run_id']} by {h['promoted_by']} — {h['reason']}")
    return True


def status(experiment_name: str):
    """Show current deployment status."""
    reg = ModelRegistry(str(DATA_DIR / "registry.db"))
    prod = reg.get_production_run(experiment_name)
    if not prod:
        print(f"⚠️  No production model for {experiment_name}")
        return
    print(f"📦 {experiment_name}")
    print(f"   Production: run {prod['id']} — v{prod['version']}")
    print(f"   Metrics: {json.dumps(prod.get('metrics', {}), indent=4)}")
    print(f"   Tags: {json.dumps(prod.get('tags', {}), indent=4)}")

    history = reg.get_promotion_history(experiment_name)
    print(f"\n   Recent promotions:")
    for h in history[:5]:
        print(f"     {h['promoted_at']}: → run {h['to_run_id']} by {h['promoted_by']}")


def main():
    parser = argparse.ArgumentParser(description="MLOps Deployment Manager")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("promote", help="Promote a model to production")
    p.add_argument("experiment", help="Experiment name")
    p.add_argument("--to-run", type=int, required=True, help="Run ID to promote")
    p.add_argument("--reason", required=True, help="Promotion reason")
    p.add_argument("--by", default="user", help="Who is promoting")

    r = sub.add_parser("rollback", help="Rollback to a previous version")
    r.add_argument("experiment", help="Experiment name")
    r.add_argument("--to-version", required=True, help="Version to rollback to")
    r.add_argument("--reason", required=True, help="Rollback reason")

    s = sub.add_parser("status", help="Show deployment status")
    s.add_argument("experiment", help="Experiment name")

    args = parser.parse_args()

    if args.command == "promote":
        promote(args.experiment, args.to_run, args.reason, args.by)
    elif args.command == "rollback":
        rollback(args.experiment, args.to_version, args.reason)
    elif args.command == "status":
        status(args.experiment)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
