import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TargetSettings:
    instrument_type: str = "EQUITY"
    region: str = "EUR"
    universe: str = "TOP2500"
    delay: int = 1


@dataclass
class FilterThresholds:
    min_sharpe: float = 1.8
    min_fitness: float = 1.3
    min_turnover: float = 0.01
    max_turnover: float = 0.4
    max_prod_correlation: float = 0.55
    max_self_correlation: float = 0.55
    require_prod_pass: bool = False


def find_check(alpha: dict, name: str) -> Tuple[Optional[str], Any]:
    for check in alpha.get("is", {}).get("checks", []) or []:
        if check.get("name") == name:
            return check.get("result"), check.get("value")
    return None, None


def get_code(alpha: dict) -> str:
    return (alpha.get("regular", {}) or {}).get("code") or ""


def get_vals(alpha: dict) -> Tuple[Any, Any]:
    is_data = alpha.get("is", {}) or {}
    return is_data.get("sharpe"), is_data.get("fitness")


def get_turnover(alpha: dict) -> Any:
    return (alpha.get("is", {}) or {}).get("turnover")


def matches_target(alpha: Dict[str, Any], target: TargetSettings) -> bool:
    settings = alpha.get("settings", {}) or {}
    return (
        settings.get("instrumentType") == target.instrument_type
        and settings.get("region") == target.region
        and settings.get("universe") == target.universe
        and settings.get("delay") == target.delay
    )


def extract_alpha_summary(alpha: Dict[str, Any]) -> Dict[str, Any]:
    sharpe, fitness = get_vals(alpha)
    turnover = get_turnover(alpha)
    cw_res, cw_val = find_check(alpha, "CONCENTRATED_WEIGHT")
    ladder_res, ladder_val = find_check(alpha, "IS_LADDER_SHARPE")
    low_res, low_val = find_check(alpha, "LOW_SUB_UNIVERSE_SHARPE")
    prod_res, prod_val = find_check(alpha, "PROD_CORRELATION")
    self_res, self_val = find_check(alpha, "SELF_CORRELATION")
    low2y_res, low2y_val = find_check(alpha, "LOW_2Y_SHARPE")
    return {
        "id": alpha.get("id"),
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        "operatorCount": ((alpha.get("regular", {}) or {}).get("operatorCount")),
        "checks": {
            "CONCENTRATED_WEIGHT": {"result": cw_res, "value": cw_val},
            "IS_LADDER_SHARPE": {"result": ladder_res, "value": ladder_val},
            "LOW_2Y_SHARPE": {"result": low2y_res, "value": low2y_val},
            "LOW_SUB_UNIVERSE_SHARPE": {"result": low_res, "value": low_val},
            "PROD_CORRELATION": {"result": prod_res, "value": prod_val},
            "SELF_CORRELATION": {"result": self_res, "value": self_val},
        },
        "code": get_code(alpha),
    }


def summarize_alphas(
    results: Iterable[Dict[str, Any]],
    target: TargetSettings,
    thresholds: FilterThresholds,
    focus_id: Optional[str] = None,
) -> Dict[str, Any]:
    def stability_gate(summary: Dict[str, Any]) -> Tuple[Optional[str], Any]:
        low2y = summary["checks"]["LOW_2Y_SHARPE"]
        ladder = summary["checks"]["IS_LADDER_SHARPE"]
        if low2y["result"] is not None:
            return low2y["result"], low2y["value"]
        return ladder["result"], ladder["value"]

    stats = {
        "cw_pass": 0,
        "ladder_pass": 0,
        "lowsub_pass": 0,
        "all_three_pass": 0,
        "cw_pending": 0,
        "ladder_pending": 0,
        "lowsub_pending": 0,
        "cw_fail": 0,
        "ladder_fail": 0,
        "lowsub_fail": 0,
    }
    passed: List[Dict[str, Any]] = []
    ladder_debug: List[Dict[str, Any]] = []
    ladder_pass_debug: List[Dict[str, Any]] = []
    cw_only_ladder_debug: List[Dict[str, Any]] = []
    target_results: List[Dict[str, Any]] = []

    for alpha in results:
        if focus_id is not None and alpha.get("id") != focus_id:
            continue
        if not matches_target(alpha, target):
            continue

        summary = extract_alpha_summary(alpha)
        target_results.append(summary)

        cw = summary["checks"]["CONCENTRATED_WEIGHT"]
        ladder = summary["checks"]["IS_LADDER_SHARPE"]
        low2y = summary["checks"]["LOW_2Y_SHARPE"]
        low_sub = summary["checks"]["LOW_SUB_UNIVERSE_SHARPE"]
        prod = summary["checks"]["PROD_CORRELATION"]
        self_corr = summary["checks"]["SELF_CORRELATION"]
        stability_res, stability_val = stability_gate(summary)

        _bump_check_stats(stats, "cw", cw["result"])
        _bump_check_stats(stats, "ladder", stability_res)
        _bump_check_stats(stats, "lowsub", low_sub["result"])

        if cw["result"] == "PASS" and stability_res == "PASS" and low_sub["result"] == "PASS":
            stats["all_three_pass"] += 1

        sharpe = summary["sharpe"]
        fitness = summary["fitness"]
        turnover = summary["turnover"]
        if sharpe is None or fitness is None or turnover is None:
            continue

        compact = " ".join(summary["code"].split())
        compact_short = compact[:200]

        if cw["result"] == "PASS" and low_sub["result"] == "PASS":
            ladder_debug.append(
                {
                    "id": summary["id"],
                    "ladder_result": stability_res,
                    "ladder_value": stability_val,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "code": compact_short,
                }
            )

        if cw["result"] == "PASS":
            cw_only_ladder_debug.append(
                {
                    "id": summary["id"],
                    "ladder_result": stability_res,
                    "ladder_value": stability_val,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "code": compact_short,
                }
            )

        if stability_res == "PASS":
            ladder_pass_debug.append(
                {
                    "id": summary["id"],
                    "cw_result": cw["result"],
                    "cw_value": cw["value"],
                    "low_sub_result": low_sub["result"],
                    "low_sub_value": low_sub["value"],
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "code": compact_short,
                }
            )

        prod_value = prod["value"]
        prod_ok = True
        if thresholds.require_prod_pass:
            prod_ok = prod["result"] == "PASS"
        if isinstance(prod_value, (int, float)):
            prod_ok = prod_ok and prod_value < thresholds.max_prod_correlation
        self_corr_value = self_corr["value"]
        self_ok = isinstance(self_corr_value, (int, float)) and self_corr_value < thresholds.max_self_correlation

        if (
            cw["result"] == "PASS"
            and stability_res == "PASS"
            and low_sub["result"] == "PASS"
            and sharpe >= thresholds.min_sharpe
            and fitness >= thresholds.min_fitness
            and thresholds.min_turnover <= turnover <= thresholds.max_turnover
            and prod_ok
            and self_ok
        ):
            passed.append(
                {
                    "id": summary["id"],
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                    "cw": cw["value"],
                    "ladder": stability_val,
                    "low_sub": low_sub["value"],
                    "prod_result": prod["result"],
                    "prod_value": prod["value"],
                    "self_result": self_corr["result"],
                    "self_value": self_corr["value"],
                    "operatorCount": summary["operatorCount"],
                    "code": compact,
                }
            )

    passed.sort(key=lambda item: (-item["sharpe"], -item["fitness"]))
    ladder_debug.sort(key=lambda item: (-_sortable_num(item["ladder_value"]), -_sortable_num(item["sharpe"])))
    ladder_pass_debug.sort(key=lambda item: (-_sortable_num(item["sharpe"]), -_sortable_num(item["fitness"])))
    cw_only_ladder_debug.sort(key=lambda item: (-_sortable_num(item["ladder_value"]),))

    return {
        "target": asdict(target),
        "thresholds": asdict(thresholds),
        "target_alpha_count": len(target_results),
        "stats": stats,
        "passed": passed,
        "ladder_debug": ladder_debug[:10],
        "ladder_pass_debug": ladder_pass_debug[:10],
        "cw_only_ladder_debug": cw_only_ladder_debug[:10],
    }


def _bump_check_stats(stats: Dict[str, int], prefix: str, result: Optional[str]) -> None:
    if result == "PASS":
        stats[f"{prefix}_pass"] += 1
    elif result == "PENDING":
        stats[f"{prefix}_pending"] += 1
    elif result is not None:
        stats[f"{prefix}_fail"] += 1


def _sortable_num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else -1e9


def print_summary(report: Dict[str, Any]) -> None:
    print("target:", report["target"])
    print("thresholds:", report["thresholds"])
    print("target_alpha_count:", report["target_alpha_count"])
    print("stats:", report["stats"])
    print("candidates_all_three_checks_pass_with_cut:", len(report["passed"]))

    for row in report["passed"][:30]:
        print(
            f"id={row['id']} sharpe={row['sharpe']} fitness={row['fitness']} turnover={row['turnover']} "
            f"cw={row['cw']} ladder={row['ladder']} lowsub={row['low_sub']} "
            f"prod={row['prod_result']}:{row['prod_value']} self={row['self_result']}:{row['self_value']} "
            f"opCount={row['operatorCount']} code={row['code']}"
        )

    if report["ladder_debug"]:
        print("\nTop ladder candidates among cw=PASS & lowsub=PASS:")
        for row in report["ladder_debug"]:
            print(
                f"id={row['id']} ladder={row['ladder_result']}:{row['ladder_value']} "
                f"sharpe={row['sharpe']} fitness={row['fitness']} code={row['code']}"
            )

    if report["ladder_pass_debug"]:
        print("\nLADDER_SHARPE=PASS alphas:")
        for row in report["ladder_pass_debug"]:
            print(
                f"id={row['id']} sharpe={row['sharpe']} fitness={row['fitness']} "
                f"cw={row['cw_result']}:{row['cw_value']} "
                f"low_sub={row['low_sub_result']}:{row['low_sub_value']} code={row['code']}"
            )

    if report["cw_only_ladder_debug"]:
        print("\nTop IS_LADDER_SHARPE values among CONCENTRATED_WEIGHT=PASS:")
        for row in report["cw_only_ladder_debug"]:
            print(
                f"id={row['id']} ladder={row['ladder_result']}:{row['ladder_value']} "
                f"sharpe={row['sharpe']} fitness={row['fitness']} code={row['code']}"
            )


def save_summary(path: str, report: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def load_results(path: str) -> List[Dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj.get("results", []) or []


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python filter_wq_alphas.py <json_file> [min_sharpe] [min_fitness] "
            "[focus_id] [region] [universe] [delay]"
        )
        raise SystemExit(2)

    path = sys.argv[1]
    min_sharpe = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.8
    min_fitness = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.3
    focus_id = sys.argv[4] if len(sys.argv) >= 5 else None
    region = sys.argv[5] if len(sys.argv) >= 6 else "EUR"
    universe = sys.argv[6] if len(sys.argv) >= 7 else "TOP2500"
    delay = int(sys.argv[7]) if len(sys.argv) >= 8 else 1

    report = summarize_alphas(
        load_results(path),
        target=TargetSettings(region=region, universe=universe, delay=delay),
        thresholds=FilterThresholds(min_sharpe=min_sharpe, min_fitness=min_fitness),
        focus_id=focus_id,
    )
    print_summary(report)


if __name__ == "__main__":
    main()
