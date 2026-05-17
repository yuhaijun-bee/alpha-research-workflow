from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import requests
from requests.auth import HTTPBasicAuth


API_BASE_URL = "https://api.worldquantbrain.com"


def load_credentials(path: str = "credential.txt") -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"email": raw[0], "password": raw[1]}


def fetch_datafields_via_api(
    *,
    instrument_type: str,
    region: str,
    universe: str,
    delay: int,
    search: str | None = None,
    dataset_id: str | None = None,
    data_type: str | None = None,
    theme: str | None = None,
    credentials_path: str = "credential.txt",
    timeout: int = 60,
) -> Dict[str, Any]:
    creds = load_credentials(credentials_path)
    params = {
        "instrumentType": instrument_type,
        "region": region,
        "universe": universe,
        "delay": delay,
    }
    if search:
        params["search"] = search
    if dataset_id:
        params["dataset.id"] = dataset_id
    if data_type:
        params["type"] = data_type
    if theme:
        params["theme"] = theme

    response = requests.get(
        f"{API_BASE_URL}/data-fields",
        params=params,
        auth=HTTPBasicAuth(creds["email"], creds["password"]),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
