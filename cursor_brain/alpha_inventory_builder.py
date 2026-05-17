from __future__ import annotations

"""
Local alpha inventory utilities.

Preferred path:
- Use MCP tools for platform reads.
- Use this module primarily for local record shaping and local inventory writes.

Deprecated for normal workflow:
- Direct API authentication/fetch helpers in this file can still work, but they are
  more likely to trigger platform rate limits than the long-lived MCP server.
"""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from expression_fingerprint import fingerprint_expression


API_BASE_URL = "https://api.worldquantbrain.com"


@dataclass
class InventoryConfig:
    root_dir: str = "alpha_inventory"
    stages: List[str] = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.stages is None:
            self.stages = ["OS"]


def load_credentials(path: str = "credential.txt") -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"email": raw[0], "password": raw[1]}


def authenticate_session(email: str, password: str) -> requests.Session:
    session = requests.Session()
    response = session.post(
        f"{API_BASE_URL}/authentication",
        json={"email": email, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    return session


def fetch_stage_alphas(
    session: requests.Session,
    stage: str,
    limit: int = 100,
    order: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if order is None:
        order = "-dateSubmitted" if stage == "OS" else "-dateCreated"

    offset = 0
    results: List[Dict[str, Any]] = []
    while True:
        response = session.get(
            f"{API_BASE_URL}/users/self/alphas",
            params={
                "stage": stage,
                "limit": limit,
                "offset": offset,
                "order": order,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("results", []) or []
        results.extend(page)
        if not payload.get("next"):
            break
        offset += limit
    return results


def alpha_record(alpha: Dict[str, Any]) -> Dict[str, Any]:
    code = ((alpha.get("regular", {}) or {}).get("code")) or ""
    fingerprint = asdict(fingerprint_expression(code)) if code else {}
    record = {
        "id": alpha.get("id"),
        "stage": alpha.get("stage"),
        "type": alpha.get("type"),
        "author": alpha.get("author"),
        "name": alpha.get("name"),
        "dateCreated": alpha.get("dateCreated"),
        "dateSubmitted": alpha.get("dateSubmitted"),
        "dateModified": alpha.get("dateModified"),
        "status": alpha.get("status"),
        "settings": alpha.get("settings"),
        "expression": code,
        "operatorCount": ((alpha.get("regular", {}) or {}).get("operatorCount")),
        "fields": fingerprint.get("fields", []),
        "operators": fingerprint.get("operators", []),
        "themes": fingerprint.get("themes", []),
        "theme_operators": fingerprint.get("theme_operators", []),
        "common_operator_hits": fingerprint.get("common_operator_hits", []),
        "skeleton": fingerprint.get("skeleton"),
        "max_depth": fingerprint.get("max_depth"),
        "checks": ((alpha.get("is", {}) or {}).get("checks", [])),
        "is_metrics": alpha.get("is"),
        "os_metrics": alpha.get("os"),
        "train_metrics": alpha.get("train"),
        "test_metrics": alpha.get("test"),
        "classifications": alpha.get("classifications", []),
        "tags": alpha.get("tags", []),
        "hidden": alpha.get("hidden"),
    }
    return record


def _record_sort_key(record: Dict[str, Any]) -> tuple[str, str]:
    return (record.get("dateSubmitted") or "", record.get("dateCreated") or "")


def _load_stage_records(stage_dir: Path) -> List[Dict[str, Any]]:
    records_dir = stage_dir / "records"
    if not records_dir.is_dir():
        return []

    records: List[Dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    records.sort(key=_record_sort_key, reverse=True)
    return records


def _build_stage_metadata(stage: str, records: Sequence[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    stage_field_counter: Counter[str] = Counter()
    stage_region_counter: Counter[str] = Counter()

    for record in records:
        stage_field_counter.update(record.get("fields", []))
        region = ((record.get("settings") or {}).get("region")) or "UNKNOWN"
        stage_region_counter[region] += 1

    stage_index = {
        "stage": stage,
        "count": len(records),
        "generatedAt": _utc_now(),
        "records": list(records),
    }
    stage_fields = {
        "stage": stage,
        "generatedAt": _utc_now(),
        "field_usage": [
            {"field": field, "count": count}
            for field, count in stage_field_counter.most_common()
        ],
        "region_usage": [
            {"region": region, "count": count}
            for region, count in stage_region_counter.most_common()
        ],
    }
    stage_summary = {
        "count": len(records),
        "top_fields": stage_fields["field_usage"][:50],
        "regions": stage_fields["region_usage"],
    }
    return stage_index, stage_fields, stage_summary


def rebuild_inventory_metadata(root_dir: str) -> Dict[str, Any]:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    overall_field_counter: Counter[str] = Counter()
    overall_region_counter: Counter[str] = Counter()
    stage_summaries: Dict[str, Any] = {}

    for stage_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        stage = stage_dir.name
        records = _load_stage_records(stage_dir)
        stage_index, stage_fields, stage_summary = _build_stage_metadata(stage, records)

        (stage_dir / "index.json").write_text(
            json.dumps(stage_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (stage_dir / "field_usage.json").write_text(
            json.dumps(stage_fields, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        stage_summaries[stage] = stage_summary
        for record in records:
            overall_field_counter.update(record.get("fields", []))
            region = ((record.get("settings") or {}).get("region")) or "UNKNOWN"
            overall_region_counter[region] += 1

    overall_summary = {
        "generatedAt": _utc_now(),
        "stages": stage_summaries,
        "overall_field_usage": [
            {"field": field, "count": count}
            for field, count in overall_field_counter.most_common()
        ],
        "overall_region_usage": [
            {"region": region, "count": count}
            for region, count in overall_region_counter.most_common()
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(overall_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return overall_summary


def build_inventory(
    session: requests.Session,
    config: InventoryConfig,
) -> Dict[str, Any]:
    root = Path(config.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    overall_field_counter: Counter[str] = Counter()
    overall_region_counter: Counter[str] = Counter()
    stage_summaries: Dict[str, Any] = {}

    for stage in config.stages:
        alphas = fetch_stage_alphas(session, stage=stage, limit=config.limit)
        stage_dir = root / stage
        records_dir = stage_dir / "records"
        records_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []

        for alpha in alphas:
            record = alpha_record(alpha)
            records.append(record)
            overall_field_counter.update(record["fields"])
            region = ((record.get("settings") or {}).get("region")) or "UNKNOWN"
            overall_region_counter[region] += 1

            (records_dir / f"{record['id']}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        records.sort(key=_record_sort_key, reverse=True)
        stage_index, stage_fields, stage_summary = _build_stage_metadata(stage, records)

        (stage_dir / "index.json").write_text(
            json.dumps(stage_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (stage_dir / "field_usage.json").write_text(
            json.dumps(stage_fields, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        stage_summaries[stage] = stage_summary

    overall_summary = {
        "generatedAt": _utc_now(),
        "stages": stage_summaries,
        "overall_field_usage": [
            {"field": field, "count": count}
            for field, count in overall_field_counter.most_common()
        ],
        "overall_region_usage": [
            {"region": region, "count": count}
            for region, count in overall_region_counter.most_common()
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(overall_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return overall_summary


def update_inventory_with_alpha(
    inventory_root: str,
    alpha: Dict[str, Any],
) -> None:
    root = Path(inventory_root)
    root.mkdir(parents=True, exist_ok=True)
    stage = alpha.get("stage") or "UNKNOWN"
    stage_dir = root / stage
    records_dir = stage_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    record = alpha_record(alpha)
    (records_dir / f"{record['id']}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rebuild_inventory_metadata(inventory_root)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a local inventory of WQ BRAIN alphas.")
    parser.add_argument("--root-dir", default="alpha_inventory")
    parser.add_argument("--stages", nargs="+", default=["OS"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--credentials", default="credential.txt")
    args = parser.parse_args()

    creds = load_credentials(args.credentials)
    session = authenticate_session(creds["email"], creds["password"])
    summary = build_inventory(
        session,
        InventoryConfig(root_dir=args.root_dir, stages=args.stages, limit=args.limit),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
