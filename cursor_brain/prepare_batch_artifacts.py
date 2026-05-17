from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "batch"


def build_batch_dir(artifact_root: str, round_name: str, multisim_id: str) -> Path:
    return Path(artifact_root) / f"{_safe_name(round_name)}__{_safe_name(multisim_id)}"


def ensure_batch_layout(artifact_root: str, round_name: str, multisim_id: str) -> Dict[str, str]:
    batch_dir = build_batch_dir(artifact_root, round_name, multisim_id)
    alpha_dir = batch_dir / "alpha_details"
    batch_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "batch_dir": str(batch_dir),
        "multisim_children_json": str(batch_dir / "multisim_children.json"),
        "batch_payload_json": str(batch_dir / "batch_payload.json"),
        "manifest_json": str(batch_dir / "manifest.json"),
        "workflow_config_json": str(batch_dir / "workflow_config.json"),
        "alpha_details_dir": str(alpha_dir),
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "multisim_id": multisim_id,
        "expected_paths": paths,
        "notes": [
            "Save the raw get_multisim_children output to multisim_children.json.",
            "Save each get_alpha_details output to alpha_details/<alpha_id>.json.",
            "Save the workflow config used for this batch to workflow_config.json.",
            "Then run run_batch_postprocess.py against this directory.",
        ],
    }
    Path(paths["manifest_json"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def write_workflow_config(config_payload: Dict[str, Any], target_path: str) -> str:
    Path(target_path).write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target_path


def copy_multisim_json(source_path: str, target_path: str) -> None:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    Path(target_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_alpha_detail_json(source_path: str, alpha_details_dir: str) -> str:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    alpha_id = payload.get("id") or "unknown_alpha"
    target = Path(alpha_details_dir) / f"{_safe_name(alpha_id)}.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create and populate a standard artifact layout for one multisim batch.")
    parser.add_argument("--artifact-root", required=True, help="Root directory for batch artifacts.")
    parser.add_argument("--round-name", required=True, help="Round label.")
    parser.add_argument("--multisim-id", required=True, help="Multisim id.")
    parser.add_argument("--copy-multisim-json", help="Optional path to a saved get_multisim_children JSON file.")
    parser.add_argument("--copy-workflow-config-json", help="Optional path to a workflow config JSON file.")
    parser.add_argument(
        "--copy-alpha-detail-json",
        action="append",
        default=[],
        help="Optional path to a saved get_alpha_details JSON file. Repeatable.",
    )
    args = parser.parse_args()

    paths = ensure_batch_layout(
        artifact_root=args.artifact_root,
        round_name=args.round_name,
        multisim_id=args.multisim_id,
    )

    copied_alpha_files: List[str] = []
    if args.copy_multisim_json:
        copy_multisim_json(args.copy_multisim_json, paths["multisim_children_json"])
    if args.copy_workflow_config_json:
        payload = json.loads(Path(args.copy_workflow_config_json).read_text(encoding="utf-8"))
        write_workflow_config(payload, paths["workflow_config_json"])
    for alpha_json in args.copy_alpha_detail_json:
        copied_alpha_files.append(copy_alpha_detail_json(alpha_json, paths["alpha_details_dir"]))

    result: Dict[str, Any] = {
        "artifact_paths": paths,
        "copied_multisim_json": bool(args.copy_multisim_json),
        "copied_workflow_config_json": bool(args.copy_workflow_config_json),
        "copied_alpha_files": copied_alpha_files,
        "next_step": {
            "command": (
                "python run_batch_postprocess.py "
                f"--multisim-children-json {paths['multisim_children_json']} "
                f"--alpha-details {paths['alpha_details_dir']} "
                f"--round-name {args.round_name}"
            )
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
