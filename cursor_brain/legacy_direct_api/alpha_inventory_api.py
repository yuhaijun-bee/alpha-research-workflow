from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


API_BASE_URL = "https://api.worldquantbrain.com"


@dataclass
class InventoryApiConfig:
    credentials_path: str = "credential.txt"
    timeout: int = 60


def load_credentials(path: str = "credential.txt") -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"email": raw[0], "password": raw[1]}


def authenticate_session(email: str, password: str, timeout: int = 60) -> requests.Session:
    session = requests.Session()
    response = session.post(
        f"{API_BASE_URL}/authentication",
        json={"email": email, "password": password},
        timeout=timeout,
    )
    response.raise_for_status()
    return session


def fetch_stage_alphas(
    session: requests.Session,
    stage: str,
    limit: int = 100,
    order: Optional[str] = None,
    timeout: int = 60,
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
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("results", []) or []
        results.extend(page)
        if not payload.get("next"):
            break
        offset += limit
    return results
