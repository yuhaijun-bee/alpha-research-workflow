from __future__ import annotations

"""
Field catalog builder.

Preferred path:
- Build catalogs from exported JSON or MCP-returned payloads saved locally.

Deprecated for normal workflow:
- The live direct API path in this file is retained as a fallback, but it is not the
  recommended route because it is more exposed to platform 429 limits.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests.auth import HTTPBasicAuth


API_BASE_URL = "https://api.worldquantbrain.com"


@dataclass
class FieldCatalogRequest:
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    search: Optional[str] = None
    dataset_id: Optional[str] = None
    data_type: Optional[str] = None
    theme: Optional[str] = None

    def filename(self) -> str:
        return f"{self.region}_{self.universe}_D{self.delay}.csv"


def load_credentials(path: str = "credential.txt") -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"email": raw[0], "password": raw[1]}


def fetch_datafields_via_api(
    request: FieldCatalogRequest,
    credentials_path: str = "credential.txt",
) -> Dict[str, Any]:
    creds = load_credentials(credentials_path)
    params = {
        "instrumentType": request.instrument_type,
        "region": request.region,
        "universe": request.universe,
        "delay": request.delay,
    }
    if request.search:
        params["search"] = request.search
    if request.dataset_id:
        params["dataset.id"] = request.dataset_id
    if request.data_type:
        params["type"] = request.data_type
    if request.theme:
        params["theme"] = request.theme

    response = requests.get(
        f"{API_BASE_URL}/data-fields",
        params=params,
        auth=HTTPBasicAuth(creds["email"], creds["password"]),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def load_datafields_payload(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_datafields(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            rows = payload["results"]
        elif isinstance(payload.get("result"), dict) and isinstance(payload["result"].get("results"), list):
            rows = payload["result"]["results"]
        elif isinstance(payload.get("result"), list):
            rows = payload["result"]
        else:
            rows = []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        field_id = row.get("id") or row.get("field") or row.get("name")
        field_type = row.get("type") or row.get("dataType") or row.get("category")
        description = row.get("description") or row.get("desc") or ""
        dataset = row.get("dataset", {})
        normalized.append(
            {
                "id": field_id,
                "type": str(field_type).upper() if field_type is not None else "",
                "description": description,
                "dataset_id": dataset.get("id") if isinstance(dataset, dict) else row.get("dataset_id", ""),
                "dataset_name": dataset.get("name") if isinstance(dataset, dict) else row.get("dataset_name", ""),
            }
        )
    return [row for row in normalized if row["id"]]


def write_field_catalog_csv(rows: Iterable[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "type", "description", "dataset_id", "dataset_name"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_field_catalog_from_payload(payload: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    rows = normalize_datafields(payload)
    write_field_catalog_csv(rows, output_path)
    return {
        "output_path": output_path,
        "field_count": len(rows),
        "sample_fields": rows[:10],
    }


def build_field_catalog(
    request: FieldCatalogRequest,
    output_dir: str = "field_catalogs",
    credentials_path: str = "credential.txt",
) -> Dict[str, Any]:
    payload = fetch_datafields_via_api(request, credentials_path=credentials_path)
    output_path = str(Path(output_dir) / request.filename())
    summary = build_field_catalog_from_payload(payload, output_path)
    summary["request"] = request.__dict__
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build local field catalog CSV files.")
    parser.add_argument("--region", default="USA")
    parser.add_argument("--universe", default="TOP3000")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--instrument-type", default="EQUITY")
    parser.add_argument("--search")
    parser.add_argument("--dataset-id")
    parser.add_argument("--data-type")
    parser.add_argument("--theme")
    parser.add_argument("--output-dir", default="field_catalogs")
    parser.add_argument("--credentials", default="credential.txt")
    parser.add_argument("--input-json", help="Build from an existing get_datafields JSON export instead of live API.")
    args = parser.parse_args()

    request = FieldCatalogRequest(
        instrument_type=args.instrument_type,
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        search=args.search,
        dataset_id=args.dataset_id,
        data_type=args.data_type,
        theme=args.theme,
    )

    if args.input_json:
        payload = load_datafields_payload(args.input_json)
        summary = build_field_catalog_from_payload(
            payload,
            str(Path(args.output_dir) / request.filename()),
        )
        summary["request"] = request.__dict__
    else:
        summary = build_field_catalog(
            request,
            output_dir=args.output_dir,
            credentials_path=args.credentials,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
