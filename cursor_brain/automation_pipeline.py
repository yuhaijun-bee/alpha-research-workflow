from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from prepare_batch_artifacts import ensure_batch_layout, write_workflow_config
from workflow_runner import WorkflowConfig, WorkflowRunner, load_workflow_config, save_workflow_snapshot


ROOT = Path(__file__).parent


def _load_config(config_path: Optional[str]) -> WorkflowConfig:
    return load_workflow_config(config_path) if config_path else WorkflowConfig()


def _config_payload(config_path: Optional[str]) -> Dict[str, Any]:
    config = _load_config(config_path)
    return asdict(config)


def _resolve_log_path(config: WorkflowConfig, override: Optional[str]) -> Optional[str]:
    return override or config.automation.research_memory_path


def _discover_batch_config_path(batch_dir: Path) -> Optional[str]:
    candidate = batch_dir / "workflow_config.json"
    if candidate.is_file():
        return str(candidate)

    manifest = batch_dir / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        config_path = ((payload.get("expected_paths") or {}).get("workflow_config_json"))
        if isinstance(config_path, str) and Path(config_path).is_file():
            return config_path
    return None


def _read_expressions(path: str) -> List[Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            results: List[Any] = []
            for item in payload:
                if isinstance(item, dict):
                    if str(item.get("expression", "")).strip():
                        results.append(item)
                else:
                    expr = str(item).strip()
                    if expr:
                        results.append(expr)
            return results
        if isinstance(payload, dict) and "expressions" in payload:
            return [str(item).strip() for item in payload["expressions"] if str(item).strip()]
        if isinstance(payload, dict) and "candidates" in payload:
            return [
                item for item in payload["candidates"]
                if isinstance(item, dict) and str(item.get("expression", "")).strip()
            ]
        raise ValueError("JSON expressions file must be a list or contain an 'expressions' or 'candidates' key.")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _batch_dirs(artifact_root: str) -> Iterable[Path]:
    root = Path(artifact_root)
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


def _discover_pending_batches(artifact_root: str) -> List[Path]:
    pending: List[Path] = []
    for batch_dir in _batch_dirs(artifact_root):
        multisim = batch_dir / "multisim_children.json"
        alpha_dir = batch_dir / "alpha_details"
        processed = batch_dir / "postprocess_status.json"
        if multisim.is_file() and alpha_dir.is_dir() and not processed.is_file():
            pending.append(batch_dir)
    return pending


def _discover_round_name(batch_dir: Path) -> str:
    manifest = batch_dir / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("round_name"):
            return str(payload["round_name"])
    name = batch_dir.name
    if "__" in name:
        return name.split("__", 1)[0]
    return name


def _run_postprocess(
    *,
    batch_dir: str,
    round_name: str,
    config_path: Optional[str],
    log_path: Optional[str],
    artifact_root: Optional[str],
    skip_inventory_update: bool,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "run_batch_postprocess.py"),
        "--batch-dir",
        batch_dir,
        "--round-name",
        round_name,
    ]
    if config_path:
        command.extend(["--config", config_path])
    if log_path:
        command.extend(["--log-path", log_path])
    if artifact_root:
        command.extend(["--artifact-root", artifact_root])
    if skip_inventory_update:
        command.append("--skip-inventory-update")

    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def cmd_snapshot(args: Any) -> Dict[str, Any]:
    config = _load_config(args.config)
    runner = WorkflowRunner(config=config)
    snapshot = runner.build_workflow_snapshot(best_sharpe=args.best_sharpe, best_fitness=args.best_fitness)
    if args.snapshot_out:
        save_workflow_snapshot(args.snapshot_out, snapshot)
    return snapshot


def cmd_preflight(args: Any) -> Dict[str, Any]:
    config = _load_config(args.config)
    runner = WorkflowRunner(config=config)
    expressions = _read_expressions(args.expressions)
    report = runner.preflight_batch(expressions)
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def cmd_init_batch(args: Any) -> Dict[str, Any]:
    paths = ensure_batch_layout(args.artifact_root, args.round_name, args.multisim_id)
    config_payload = _config_payload(args.config)
    write_workflow_config(config_payload, paths["workflow_config_json"])
    return {"artifact_paths": paths, "workflow_config_json": paths["workflow_config_json"]}


def cmd_process_batch(args: Any) -> Dict[str, Any]:
    config = _load_config(args.config)
    batch_dir = Path(args.batch_dir)
    round_name = args.round_name or _discover_round_name(batch_dir)
    config_path = args.config or _discover_batch_config_path(batch_dir)
    return _run_postprocess(
        batch_dir=args.batch_dir,
        round_name=round_name,
        config_path=config_path,
        log_path=_resolve_log_path(config, args.log_path),
        artifact_root=args.artifact_root,
        skip_inventory_update=args.skip_inventory_update,
    )


def cmd_scan_pending(args: Any) -> Dict[str, Any]:
    config = _load_config(args.config)
    pending = _discover_pending_batches(args.artifact_root)
    processed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for batch_dir in pending:
        round_name = _discover_round_name(batch_dir)
        try:
            config_path = args.config or _discover_batch_config_path(batch_dir)
            payload = _run_postprocess(
                batch_dir=str(batch_dir),
                round_name=round_name,
                config_path=config_path,
                log_path=_resolve_log_path(config, args.log_path),
                artifact_root=args.artifact_root,
                skip_inventory_update=args.skip_inventory_update,
            )
            processed.append(
                {
                    "batch_dir": str(batch_dir),
                    "round_name": round_name,
                    "best_candidate": payload.get("best_candidate"),
                    "postprocess_status_json": payload.get("postprocess_status_json"),
                }
            )
        except subprocess.CalledProcessError as exc:
            errors.append(
                {
                    "batch_dir": str(batch_dir),
                    "round_name": round_name,
                    "returncode": exc.returncode,
                    "stderr": exc.stderr[-2000:] if exc.stderr else "",
                }
            )

    return {
        "artifact_root": args.artifact_root,
        "pending_count": len(pending),
        "processed_count": len(processed),
        "error_count": len(errors),
        "processed": processed,
        "errors": errors,
    }


def cmd_memory_summary(args: Any) -> Dict[str, Any]:
    config = _load_config(args.config)
    runner = WorkflowRunner(config=config)
    log_path = _resolve_log_path(config, args.log_path)
    if not log_path:
        raise SystemExit("A log path is required via --log-path or config.automation.research_memory_path")
    return runner.summarize_research_memory(log_path)


def cmd_backfill_memory(args: Any) -> Dict[str, Any]:
    config = _load_config(args.config)
    runner = WorkflowRunner(config=config)
    log_path = _resolve_log_path(config, args.log_path)
    if not log_path:
        raise SystemExit("A log path is required via --log-path or config.automation.research_memory_path")
    return runner.backfill_research_memory_log(
        log_path=log_path,
        output_path=args.output_path,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Automation entrypoint for the local alpha mining pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Generate workflow snapshot.")
    snapshot.add_argument("--config")
    snapshot.add_argument("--best-sharpe", type=float)
    snapshot.add_argument("--best-fitness", type=float)
    snapshot.add_argument("--snapshot-out")

    preflight = subparsers.add_parser("preflight", help="Run preflight against an expressions file.")
    preflight.add_argument("--config")
    preflight.add_argument("--expressions", required=True, help="Path to txt/json file of expressions.")
    preflight.add_argument("--report-out")

    init_batch = subparsers.add_parser("init-batch", help="Create a standard artifact directory for one batch.")
    init_batch.add_argument("--config")
    init_batch.add_argument("--artifact-root", required=True)
    init_batch.add_argument("--round-name", required=True)
    init_batch.add_argument("--multisim-id", required=True)

    process_batch = subparsers.add_parser("process-batch", help="Post-process one standard batch directory.")
    process_batch.add_argument("--config")
    process_batch.add_argument("--batch-dir", required=True)
    process_batch.add_argument("--round-name")
    process_batch.add_argument("--log-path")
    process_batch.add_argument("--artifact-root")
    process_batch.add_argument("--skip-inventory-update", action="store_true")

    scan_pending = subparsers.add_parser("scan-pending", help="Scan artifact root and post-process all pending batches.")
    scan_pending.add_argument("--config")
    scan_pending.add_argument("--artifact-root", required=True)
    scan_pending.add_argument("--log-path")
    scan_pending.add_argument("--skip-inventory-update", action="store_true")

    memory_summary = subparsers.add_parser("memory-summary", help="Summarize research memory log.")
    memory_summary.add_argument("--config")
    memory_summary.add_argument("--log-path", help="Optional override for research memory log path.")

    backfill_memory = subparsers.add_parser("backfill-memory", help="Backfill research memory into a legacy JSONL log.")
    backfill_memory.add_argument("--config")
    backfill_memory.add_argument("--log-path", help="Optional override for source log path.")
    backfill_memory.add_argument("--output-path", help="Optional output path. Defaults to in-place update.")

    args = parser.parse_args()

    if args.command == "snapshot":
        result = cmd_snapshot(args)
    elif args.command == "preflight":
        result = cmd_preflight(args)
    elif args.command == "init-batch":
        result = cmd_init_batch(args)
    elif args.command == "process-batch":
        result = cmd_process_batch(args)
    elif args.command == "scan-pending":
        result = cmd_scan_pending(args)
    elif args.command == "memory-summary":
        result = cmd_memory_summary(args)
    elif args.command == "backfill-memory":
        result = cmd_backfill_memory(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
