from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from workflow_runner import WorkflowConfig, WorkflowRunner, load_workflow_config


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _discover_batch_config_path(batch_dir: Path) -> str | None:
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


def _load_alpha_detail_files(paths: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in paths:
        entry = Path(path)
        if entry.is_dir():
            for child in sorted(entry.glob("*.json")):
                results.append(json.loads(child.read_text(encoding="utf-8")))
        else:
            results.append(json.loads(entry.read_text(encoding="utf-8")))
    return results


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "batch"


def _artifact_batch_dir(artifact_root: str, round_name: str, multisim_payload: Dict[str, Any]) -> Path:
    root = multisim_payload.get("result", multisim_payload)
    multisim_id = root.get("multisim_id") or root.get("id") or "unknown_multisim"
    batch_name = f"{_safe_name(round_name)}__{_safe_name(multisim_id)}"
    return Path(artifact_root) / batch_name


def _write_standard_artifacts(
    artifact_root: str,
    round_name: str,
    multisim_payload: Dict[str, Any],
    platform_alphas: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> Dict[str, str]:
    batch_dir = _artifact_batch_dir(artifact_root, round_name, multisim_payload)
    alpha_dir = batch_dir / "alpha_details"
    batch_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    multisim_path = batch_dir / "multisim_children.json"
    payload_path = batch_dir / "batch_payload.json"
    manifest_path = batch_dir / "manifest.json"

    multisim_path.write_text(json.dumps(multisim_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    alpha_files: List[str] = []
    for alpha in platform_alphas:
        alpha_id = alpha.get("id") or "unknown_alpha"
        target = alpha_dir / f"{_safe_name(alpha_id)}.json"
        target.write_text(json.dumps(alpha, indent=2, ensure_ascii=False), encoding="utf-8")
        alpha_files.append(str(target))

    manifest = {
        "round_name": round_name,
        "multisim_id": (multisim_payload.get("result", multisim_payload).get("multisim_id") or multisim_payload.get("id")),
        "generated_artifacts": {
            "multisim_children_json": str(multisim_path),
            "batch_payload_json": str(payload_path),
            "alpha_detail_files": alpha_files,
        },
        "summary": {
            "matched_alpha_count": payload.get("matched_alpha_count"),
            "missing_alpha_ids": payload.get("missing_alpha_ids", []),
            "best_candidate": payload.get("best_candidate"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "batch_dir": str(batch_dir),
        "multisim_children_json": str(multisim_path),
        "batch_payload_json": str(payload_path),
        "manifest_json": str(manifest_path),
        "alpha_details_dir": str(alpha_dir),
    }


def _write_postprocess_status(batch_dir: str, payload: Dict[str, Any]) -> str:
    status_path = Path(batch_dir) / "postprocess_status.json"
    next_round_plan = payload.get("next_round_plan", {}) or {}
    floor_winners = payload.get("floor_winners", []) or []
    target_band_winners = payload.get("target_band_winners", []) or []
    status = {
        "processed": True,
        "processed_at": payload.get("processed_at"),
        "round": payload.get("round"),
        "multisim_id": payload.get("multisim_id"),
        "matched_alpha_count": payload.get("matched_alpha_count"),
        "missing_alpha_ids": payload.get("missing_alpha_ids", []),
        "best_candidate": payload.get("best_candidate"),
        "qualified_alpha_goal": payload.get("qualified_alpha_goal"),
        "qualified_alpha_goal_met": payload.get("qualified_alpha_goal_met"),
        "qualified_alpha_count": len(floor_winners),
        "target_band_alpha_count": len(target_band_winners),
        "qualified_alpha_ids": [row.get("id") for row in floor_winners if row.get("id")],
        "target_band_alpha_ids": [row.get("id") for row in target_band_winners if row.get("id")],
        "next_step_bias": ((payload.get("batch_analysis", {}) or {}).get("next_step_bias")),
        "next_round_plan": next_round_plan,
        "family_status": next_round_plan.get("family_status"),
        "primary_goal": next_round_plan.get("primary_goal"),
        "failure_type": next_round_plan.get("failure_type"),
        "stage": next_round_plan.get("stage"),
        "stage_scope": next_round_plan.get("stage_scope"),
        "batch_region_mix": next_round_plan.get("batch_region_mix"),
        "recommended_priority_targets": next_round_plan.get("recommended_priority_targets"),
        "current_region_priority_targets": next_round_plan.get("current_region_priority_targets"),
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(status_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Post-process a multisim batch into diagnostics, logs, and inventory updates.")
    parser.add_argument("--config", help="Path to workflow config JSON.")
    parser.add_argument("--multisim-children-json", help="Path to saved get_multisim_children JSON.")
    parser.add_argument(
        "--alpha-details",
        action="append",
        default=[],
        help="Path to a get_alpha_details JSON file or a directory containing many such JSON files. Repeatable.",
    )
    parser.add_argument("--batch-dir", help="Standardized batch artifact directory containing multisim_children.json and alpha_details/.")
    parser.add_argument("--round-name", required=True, help="Round label, for example round80_else_ts_scale_breakthrough_completed.")
    parser.add_argument("--log-path", help="Append the generated batch payload to this log file.")
    parser.add_argument("--payload-out", help="Write the generated batch payload JSON to this path.")
    parser.add_argument("--artifact-root", help="Optional root directory for standardized batch artifact output.")
    parser.add_argument("--notes", action="append", default=[], help="Optional note line to include in payload. Repeatable.")
    parser.add_argument("--skip-inventory-update", action="store_true", help="Do not update local inventory records.")
    args = parser.parse_args()

    multisim_children_json = args.multisim_children_json
    alpha_detail_inputs = list(args.alpha_details)
    config_path = args.config
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not config_path:
            config_path = _discover_batch_config_path(batch_dir)
        if not multisim_children_json:
            multisim_children_json = str(batch_dir / "multisim_children.json")
        if not alpha_detail_inputs:
            alpha_detail_inputs = [str(batch_dir / "alpha_details")]

    config = load_workflow_config(config_path) if config_path else WorkflowConfig()
    runner = WorkflowRunner(config=config)

    if not multisim_children_json:
        raise SystemExit("--multisim-children-json or --batch-dir is required")
    if not alpha_detail_inputs:
        raise SystemExit("--alpha-details or --batch-dir is required")

    multisim_payload = _load_json(multisim_children_json)
    platform_alphas = _load_alpha_detail_files(alpha_detail_inputs)

    payload = runner.build_batch_payload_from_multisim(
        round_name=args.round_name,
        multisim_payload=multisim_payload,
        platform_alphas=platform_alphas,
        notes=args.notes,
    )
    payload["processed_at"] = datetime.now(timezone.utc).isoformat()
    payload["config_path_used"] = config_path

    if args.log_path:
        runner.append_backtest_results(args.log_path, payload)

    if not args.skip_inventory_update:
        runner.update_local_inventory_many(platform_alphas)

    if args.payload_out:
        Path(args.payload_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.artifact_root:
        artifact_paths = _write_standard_artifacts(
            artifact_root=args.artifact_root,
            round_name=args.round_name,
            multisim_payload=multisim_payload,
            platform_alphas=platform_alphas,
            payload=payload,
        )
        payload["artifact_paths"] = artifact_paths
        payload["postprocess_status_json"] = _write_postprocess_status(artifact_paths["batch_dir"], payload)
    elif args.batch_dir:
        payload["postprocess_status_json"] = _write_postprocess_status(args.batch_dir, payload)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
