from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


API_BASE_URL = "https://api.worldquantbrain.com"
DEFAULT_CREDENTIALS_PATH = "credential.txt"
DEFAULT_OPERATORS_JSON_PATH = "operators.json"
DEFAULT_OPERATOR_CACHE_PATH = "operator_compatibility_cache.json"

UNKNOWN_OPERATOR_PATTERN = re.compile(r'unknown operator "([^"]+)"', re.IGNORECASE)


@dataclass
class SessionConfig:
    credentials_path: str = DEFAULT_CREDENTIALS_PATH
    timeout: int = 60
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_credentials(path: str = DEFAULT_CREDENTIALS_PATH) -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"email": raw[0], "password": raw[1]}


def authenticate_session(config: SessionConfig) -> requests.Session:
    creds = load_credentials(config.credentials_path)
    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries):
        session = requests.Session()
        response = session.post(
            f"{API_BASE_URL}/authentication",
            json={"email": creds["email"], "password": creds["password"]},
            timeout=config.timeout,
        )
        if response.status_code == 429:
            session.close()
            last_error = requests.HTTPError(
                f"429 Client Error: Too Many Requests for url: {response.url}",
                response=response,
            )
            if attempt + 1 >= config.max_retries:
                raise last_error
            time.sleep(config.retry_backoff_seconds * (attempt + 1))
            continue
        response.raise_for_status()
        return session
    if last_error is not None:  # pragma: no cover - defensive path
        raise last_error
    raise RuntimeError("Authentication failed.")


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    config: SessionConfig,
    **kwargs: Any,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries):
        response = session.request(method, url, timeout=config.timeout, **kwargs)
        if response.status_code == 429 and attempt + 1 < config.max_retries:
            time.sleep(config.retry_backoff_seconds * (attempt + 1))
            continue
        try:
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - defensive path
            last_error = exc
            if attempt + 1 >= config.max_retries:
                raise
            time.sleep(config.retry_backoff_seconds * (attempt + 1))
    if last_error is not None:  # pragma: no cover - defensive path
        raise last_error
    raise RuntimeError("Failed to fetch JSON response.")


def _normalize_simulation_id(simulation_id: str) -> str:
    return simulation_id.rstrip("/").split("/")[-1]


def _simulation_url(simulation_id: str) -> str:
    return f"{API_BASE_URL}/simulations/{_normalize_simulation_id(simulation_id)}"


def _alpha_url(alpha_id: str) -> str:
    return f"{API_BASE_URL}/alphas/{alpha_id}"


def load_operator_cache(path: str = DEFAULT_OPERATOR_CACHE_PATH) -> Dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.is_file():
        return {"records": []}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_operator_cache(cache: Dict[str, Any], path: str = DEFAULT_OPERATOR_CACHE_PATH) -> None:
    Path(path).write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def record_operator_compatibility(
    *,
    operator: str,
    status: str,
    source: str,
    message: Optional[str] = None,
    region: Optional[str] = None,
    delay: Optional[int] = None,
    language: Optional[str] = None,
    cache_path: str = DEFAULT_OPERATOR_CACHE_PATH,
) -> Dict[str, Any]:
    cache = load_operator_cache(cache_path)
    records = cache.setdefault("records", [])
    entry = {
        "operator": operator,
        "status": status,
        "source": source,
        "message": message,
        "region": region,
        "delay": delay,
        "language": language,
        "timestamp": _utc_now(),
    }
    records.append(entry)
    save_operator_cache(cache, cache_path)
    return entry


def _extract_unknown_operator(message: Optional[str]) -> Optional[str]:
    if not message:
        return None
    match = UNKNOWN_OPERATOR_PATTERN.search(message)
    return match.group(1) if match else None


def _extract_alpha_id(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("alpha", "alphaId", "alpha_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value.rstrip("/").split("/")[-1]
        if isinstance(value, dict):
            candidate = value.get("id") or value.get("alpha")
            if isinstance(candidate, str) and candidate:
                return candidate.rstrip("/").split("/")[-1]
    links = payload.get("_links") or payload.get("links") or {}
    if isinstance(links, dict):
        for candidate_key in ("alpha", "result", "self"):
            candidate = links.get(candidate_key)
            if isinstance(candidate, str) and "/alphas/" in candidate:
                return candidate.rstrip("/").split("/")[-1]
    return None


def get_simulation_status(
    simulation_id: str,
    *,
    session: Optional[requests.Session] = None,
    config: Optional[SessionConfig] = None,
    cache_path: str = DEFAULT_OPERATOR_CACHE_PATH,
) -> Dict[str, Any]:
    config = config or SessionConfig()
    owned_session = session is None
    if session is None:
        session = authenticate_session(config)
    try:
        payload = request_json(session, "GET", _simulation_url(simulation_id), config=config)
    finally:
        if owned_session:
            session.close()

    message = payload.get("message")
    operator = _extract_unknown_operator(message)
    if operator:
        settings = payload.get("settings") or {}
        record_operator_compatibility(
            operator=operator,
            status="blocked",
            source="simulation_error",
            message=message,
            region=settings.get("region"),
            delay=settings.get("delay"),
            language=settings.get("language"),
            cache_path=cache_path,
        )

    result = {
        "simulation_id": _normalize_simulation_id(simulation_id),
        "status": payload.get("status"),
        "progress": payload.get("progress"),
        "alpha_id": _extract_alpha_id(payload),
        "error": message if payload.get("status") == "ERROR" else None,
        "expression": payload.get("regular"),
        "children": payload.get("children", []),
        "settings": payload.get("settings"),
        "raw": payload,
    }
    return result


def get_multisim_children(
    multisim_id: str,
    *,
    config: Optional[SessionConfig] = None,
    cache_path: str = DEFAULT_OPERATOR_CACHE_PATH,
) -> Dict[str, Any]:
    config = config or SessionConfig()
    session = authenticate_session(config)
    try:
        parent = get_simulation_status(
            multisim_id,
            session=session,
            config=config,
            cache_path=cache_path,
        )
        child_ids = parent.get("children") or []
        children: List[Dict[str, Any]] = []
        for child_id in child_ids:
            children.append(
                get_simulation_status(
                    str(child_id),
                    session=session,
                    config=config,
                    cache_path=cache_path,
                )
            )
    finally:
        session.close()

    return {
        "multisim_id": parent["simulation_id"],
        "status": parent["status"],
        "progress": parent["progress"],
        "error": parent["error"],
        "children": children,
    }


def _summarize_checks(checks: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for check in checks or []:
        name = check.get("name")
        if name:
            summary[name] = {
                "result": check.get("result"),
                "value": check.get("value"),
                "raw": check,
            }
    return summary


def get_alpha_checks_summary(
    alpha_id: str,
    *,
    config: Optional[SessionConfig] = None,
) -> Dict[str, Any]:
    config = config or SessionConfig()
    session = authenticate_session(config)
    try:
        alpha = request_json(session, "GET", _alpha_url(alpha_id), config=config)
    finally:
        session.close()

    settings = alpha.get("settings") or {}
    regular = alpha.get("regular") or {}
    is_data = alpha.get("is") or {}
    checks = is_data.get("checks") or []
    check_map = _summarize_checks(checks)

    def _pass(name: str) -> Optional[bool]:
        result = check_map.get(name, {}).get("result")
        if result is None:
            return None
        return result == "PASS"

    prod_value = check_map.get("PROD_CORRELATION", {}).get("value")

    return {
        "alpha_id": alpha.get("id") or alpha_id,
        "region": settings.get("region"),
        "universe": settings.get("universe"),
        "delay": settings.get("delay"),
        "neutralization": settings.get("neutralization"),
        "language": settings.get("language"),
        "expression": regular.get("code"),
        "operator_count": regular.get("operatorCount"),
        "sharpe": is_data.get("sharpe"),
        "fitness": is_data.get("fitness"),
        "turnover": is_data.get("turnover"),
        "sub_universe_sharpe_pass": _pass("LOW_SUB_UNIVERSE_SHARPE"),
        "is_ladder_sharpe_pass": _pass("IS_LADDER_SHARPE"),
        "concentrated_weight_pass": _pass("CONCENTRATED_WEIGHT"),
        "prod_correlation": prod_value,
        "prod_correlation_pass": _pass("PROD_CORRELATION"),
        "self_correlation": check_map.get("SELF_CORRELATION", {}).get("value"),
        "all_checks": checks,
        "raw": alpha,
    }


def get_platform_operator_whitelist(
    *,
    region: Optional[str] = None,
    delay: Optional[int] = None,
    language: Optional[str] = None,
    operators_json_path: str = DEFAULT_OPERATORS_JSON_PATH,
    cache_path: str = DEFAULT_OPERATOR_CACHE_PATH,
) -> Dict[str, Any]:
    payload = json.loads(Path(operators_json_path).read_text(encoding="utf-8"))
    operators = sorted(
        operator["name"]
        for operator in payload.get("operators", [])
        if isinstance(operator, dict) and operator.get("name")
    )

    cache = load_operator_cache(cache_path)
    blocked: List[Dict[str, Any]] = []
    for entry in cache.get("records", []):
        if entry.get("status") != "blocked":
            continue
        if region is not None and entry.get("region") not in (None, region):
            continue
        if delay is not None and entry.get("delay") not in (None, delay):
            continue
        if language is not None and entry.get("language") not in (None, language):
            continue
        blocked.append(entry)

    blocked_names = sorted({entry["operator"] for entry in blocked})
    known_allowed = [name for name in operators if name not in blocked_names]
    return {
        "region": region,
        "delay": delay,
        "language": language,
        "source": "local operators.json + observed platform incompatibility cache",
        "operators": operators,
        "known_allowed": known_allowed,
        "blocked_operators": blocked_names,
        "blocked_records": blocked,
    }


def append_optimization_log(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    timestamped = {
        "timestamp": _utc_now(),
        "payload": payload,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(timestamped, ensure_ascii=False))
        handle.write("\n")
    return {"path": path, "written": True, "timestamp": timestamped["timestamp"]}


def _print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local toolkit for WQ BRAIN workflow support.")
    parser.add_argument("--credentials", default=DEFAULT_CREDENTIALS_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    multisim = subparsers.add_parser("get-multisim-children")
    multisim.add_argument("multisim_id")
    multisim.add_argument("--operator-cache", default=DEFAULT_OPERATOR_CACHE_PATH)

    sim = subparsers.add_parser("get-simulation-status")
    sim.add_argument("simulation_id")
    sim.add_argument("--operator-cache", default=DEFAULT_OPERATOR_CACHE_PATH)

    alpha = subparsers.add_parser("get-alpha-checks-summary")
    alpha.add_argument("alpha_id")

    whitelist = subparsers.add_parser("get-platform-operator-whitelist")
    whitelist.add_argument("--region")
    whitelist.add_argument("--delay", type=int)
    whitelist.add_argument("--language")
    whitelist.add_argument("--operators-json", default=DEFAULT_OPERATORS_JSON_PATH)
    whitelist.add_argument("--operator-cache", default=DEFAULT_OPERATOR_CACHE_PATH)

    log = subparsers.add_parser("append-optimization-log")
    log.add_argument("path")
    log_group = log.add_mutually_exclusive_group(required=True)
    log_group.add_argument("--payload", help="JSON string payload.")
    log_group.add_argument("--payload-file", help="Path to JSON file payload.")

    record = subparsers.add_parser("record-operator-compatibility")
    record.add_argument("operator")
    record.add_argument("status", choices=["blocked", "allowed", "unknown"])
    record.add_argument("--source", default="manual")
    record.add_argument("--message")
    record.add_argument("--region")
    record.add_argument("--delay", type=int)
    record.add_argument("--language")
    record.add_argument("--operator-cache", default=DEFAULT_OPERATOR_CACHE_PATH)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = SessionConfig(credentials_path=args.credentials)

    if args.command == "get-multisim-children":
        _print_json(
            get_multisim_children(
                args.multisim_id,
                config=config,
                cache_path=args.operator_cache,
            )
        )
        return

    if args.command == "get-simulation-status":
        _print_json(
            get_simulation_status(
                args.simulation_id,
                config=config,
                cache_path=args.operator_cache,
            )
        )
        return

    if args.command == "get-alpha-checks-summary":
        _print_json(get_alpha_checks_summary(args.alpha_id, config=config))
        return

    if args.command == "get-platform-operator-whitelist":
        _print_json(
            get_platform_operator_whitelist(
                region=args.region,
                delay=args.delay,
                language=args.language,
                operators_json_path=args.operators_json,
                cache_path=args.operator_cache,
            )
        )
        return

    if args.command == "append-optimization-log":
        if args.payload_file:
            payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8-sig"))
        else:
            payload = json.loads(args.payload)
        _print_json(append_optimization_log(args.path, payload))
        return

    if args.command == "record-operator-compatibility":
        _print_json(
            record_operator_compatibility(
                operator=args.operator,
                status=args.status,
                source=args.source,
                message=args.message,
                region=args.region,
                delay=args.delay,
                language=args.language,
                cache_path=args.operator_cache,
            )
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
