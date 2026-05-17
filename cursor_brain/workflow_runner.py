from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from alpha_inventory_builder import update_inventory_with_alpha
from expression_fingerprint import (
    COMMON_OPERATOR_SET,
    THEME_MAP,
    compare_fingerprints,
    fingerprint_expression,
)
from expression_validator import resolve_field_catalog_path, validate_expression_detailed
from filter_wq_alphas import FilterThresholds, TargetSettings, summarize_alphas


REGION_ALIAS_MAP = {
    "JPN": "ASI",
}

STAGE_A = "Stage A"
STAGE_B = "Stage B"


@dataclass
class ResearchPolicy:
    min_paper_count: int = 50
    require_economic_rationale_before_expression: bool = True
    preferred_authority_tiers: List[str] = field(
        default_factory=lambda: [
            "Top finance journals",
            "NBER/SSRN working papers",
            "Quant platform official research",
            "Exchange/MSCI/AQR and comparable institutional research",
        ]
    )
    preferred_domains: List[str] = field(
        default_factory=lambda: [
            "journaloffinance.org",
            "academic.oup.com",
            "sciencedirect.com",
            "nber.org",
            "ssrn.com",
            "msci.com",
            "aqr.com",
            "support.worldquantbrain.com",
        ]
    )
    notes: List[str] = field(
        default_factory=lambda: [
            "Research intake must prioritize authoritative finance and quant sources before generic web content.",
            "Research notes should capture financial intuition, robustness caveats, and field-to-alpha mapping.",
            "At least 50 relevant papers/articles should be reviewed before large-scale mining begins.",
            "Favor dataset families and regions that expand pyramids and improve consultant income multipliers.",
            "Prefer economically coherent, low-correlation signals over parameter-heavy variants.",
            "When a family is plateaued, stop blind local tweaking and expand the search through literature, forum cases, and peer mining experience first.",
            "Plateau handling should explicitly compare current failure mode against prior lessons before launching another variant batch.",
            "Before writing any expression, define the economic mechanism first; do not start from operator combinations without a financial story.",
        ]
    )


@dataclass
class PerformanceTargets:
    target_sharpe: float = 2.2
    min_sharpe: float = 1.8
    target_fitness: float = 1.6
    min_fitness: float = 1.3
    target_prod_correlation: float = 0.4
    max_prod_correlation: float = 0.55
    target_self_correlation: float = 0.4
    max_self_correlation: float = 0.55
    min_turnover: float = 0.01
    max_turnover: float = 0.40
    min_margin_bps: float = 4.0
    require_stability_pass: bool = True
    require_sub_universe_pass: bool = True
    require_weight_pass: bool = True


@dataclass
class AutomationPolicy:
    single_round_goal_required: bool = True
    plateau_min_neighbor_count: int = 3
    plateau_metric_spread_threshold: float = 0.03
    plateau_improvement_threshold: float = 0.02
    min_qualified_alphas_per_day: int = 2
    prioritize_pyramids: bool = True
    prioritize_income_multipliers: bool = True
    frozen_directions: List[str] = field(default_factory=list)
    active_family: Optional[str] = None
    research_memory_path: Optional[str] = None
    experience_library_path: Optional[str] = None
    stage_scope: str = "current_quarter_only"
    region_allocation: Dict[str, float] = field(
        default_factory=lambda: {
            "EUR": 0.50,
            "MEA": 0.25,
            "ASI": 0.25,
        }
    )
    pyramid_priority_targets: List[str] = field(
        default_factory=lambda: [
            "EUR:D1:Fundamental",
            "EUR:D1:News",
            "EUR:D1:Other",
            "EUR:D1:Sentiment",
            "MEA:D1:Fundamental",
            "MEA:D1:Analyst",
            "ASI:D1:Model",
            "ASI:D1:Analyst",
            "ASI:D1:Sentiment",
            "ASI:D1:Other",
        ]
    )
    pyramid_priority_table: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"target": "EUR:D1:Fundamental", "priority": 1, "multiplier": 1.2, "difficulty": "medium_low", "role": "primary"},
            {"target": "EUR:D1:News", "priority": 2, "multiplier": 1.5, "difficulty": "high", "role": "primary"},
            {"target": "EUR:D1:Other", "priority": 3, "multiplier": 1.6, "difficulty": "high", "role": "primary"},
            {"target": "EUR:D1:Sentiment", "priority": 4, "multiplier": 1.4, "difficulty": "medium_high", "role": "primary"},
            {"target": "MEA:D1:Fundamental", "priority": 5, "multiplier": 1.5, "difficulty": "medium_high", "role": "secondary"},
            {"target": "MEA:D1:Analyst", "priority": 6, "multiplier": 1.9, "difficulty": "very_high", "role": "secondary"},
            {"target": "ASI:D1:Model", "priority": 7, "multiplier": 1.3, "difficulty": "medium", "role": "secondary"},
            {"target": "ASI:D1:Analyst", "priority": 8, "multiplier": 1.4, "difficulty": "high", "role": "secondary"},
            {"target": "ASI:D1:Sentiment", "priority": 9, "multiplier": 1.5, "difficulty": "high", "role": "secondary"},
            {"target": "ASI:D1:Other", "priority": 10, "multiplier": 1.5, "difficulty": "very_high", "role": "secondary"},
        ]
    )


@dataclass
class WorkflowConfig:
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "NONE"
    language: str = "FASTEXPR"
    field_types: List[str] = field(default_factory=list)
    baseline_alpha_id: Optional[str] = None
    baseline_expression: Optional[str] = None
    field_catalog_path: Optional[str] = None
    max_operator_count: int = 8
    batch_size: int = 8
    inventory_root: str = "alpha_inventory"
    credentials_path: str = "credential.txt"
    operator_cache_path: str = "operator_compatibility_cache.json"
    research: ResearchPolicy = field(default_factory=ResearchPolicy)
    targets: PerformanceTargets = field(default_factory=PerformanceTargets)
    automation: AutomationPolicy = field(default_factory=AutomationPolicy)


@dataclass
class CandidateBlueprint:
    slot: int
    role: str
    strategy_family: str
    focus_region: str
    required_themes: List[str]
    priority_pyramids: List[str]
    notes: List[str]


@dataclass
class CandidatePreflight:
    expression: str
    economic_rationale: Optional[str]
    fingerprint: Dict[str, Any]
    validation: Dict[str, Any]
    issues: List[str]


@dataclass
class MetricDiagnosis:
    metric: str
    status: str
    observed_value: Any
    diagnosis: str
    recommended_actions: List[str]


@dataclass
class WorkflowHeuristics:
    platform_detection_rules: List[str]
    metric_resolution_order: List[str]
    late_stage_priorities: List[str]
    final_breakthrough_patterns: List[str]


def _safe_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _normalize_expression_candidate(candidate: Any) -> Dict[str, Any]:
    if isinstance(candidate, str):
        return {
            "expression": candidate.strip(),
            "economic_rationale": None,
            "hypothesis": None,
            "dataset_family": None,
        }

    if isinstance(candidate, dict):
        expression = str(candidate.get("expression", "")).strip()
        rationale = candidate.get("economic_rationale")
        if rationale is None:
            rationale = candidate.get("rationale")
        hypothesis = candidate.get("hypothesis")
        dataset_family = candidate.get("dataset_family")
        return {
            "expression": expression,
            "economic_rationale": str(rationale).strip() if isinstance(rationale, str) else None,
            "hypothesis": str(hypothesis).strip() if isinstance(hypothesis, str) else None,
            "dataset_family": str(dataset_family).strip() if isinstance(dataset_family, str) else None,
        }

    return {
        "expression": str(candidate).strip(),
        "economic_rationale": None,
        "hypothesis": None,
        "dataset_family": None,
    }


def _infer_goal_from_text(text: str) -> Tuple[str, str]:
    lowered = text.lower()
    if "corr" in lowered or "decorrelation" in lowered or "correlation" in lowered:
        return "correlation_failure", "correlation_push"
    if "ladder" in lowered or "stability" in lowered or "2y" in lowered:
        return "temporal_stability_failure", "ladder_repair"
    if "concentration" in lowered or "tail" in lowered:
        return "concentration_failure", "concentration_repair"
    if "subu" in lowered or "sub-universe" in lowered or "subuniverse" in lowered or "breadth" in lowered or "coverage" in lowered:
        return "breadth_failure", "sub_universe_repair"
    if "fitness" in lowered:
        return "setting_failure", "fitness_push"
    if "newline" in lowered or "new family" in lowered or "new subfamily" in lowered or "rebuild" in lowered:
        return "structure_failure", "new_family_validation"
    return "structure_failure", "structure_rebuild"


def _as_mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_single_dataset_alpha(alpha_summary: Dict[str, Any]) -> bool:
    raw = _as_mapping(alpha_summary.get("raw"))
    classifications = raw.get("classifications", []) or []
    for item in classifications:
        if isinstance(item, dict) and item.get("id") == "DATA_USAGE:SINGLE_DATA_SET":
            return True
    return False


def _stability_check_name(alpha_summary: Dict[str, Any]) -> str:
    return "LOW_2Y_SHARPE" if _is_single_dataset_alpha(alpha_summary) else "IS_LADDER_SHARPE"


def _stability_check(alpha_summary: Dict[str, Any]) -> Dict[str, Any]:
    checks = _as_mapping(alpha_summary.get("checks"))
    return _as_mapping(checks.get(_stability_check_name(alpha_summary), {}))


def _stability_is_passing(alpha_summary: Dict[str, Any]) -> bool:
    return _stability_check(alpha_summary).get("result") == "PASS"


def _parse_pyramid_target(target: str) -> Optional[Dict[str, Any]]:
    parts = [part.strip() for part in str(target).split(":")]
    if len(parts) != 3:
        return None
    region, delay_token, category = parts
    delay = None
    if delay_token.upper().startswith("D"):
        try:
            delay = int(delay_token[1:])
        except ValueError:
            delay = None
    return {
        "region": region,
        "delay_token": delay_token,
        "delay": delay,
        "category": category,
        "target": f"{region}:{delay_token}:{category}",
    }


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class WorkflowRunner:
    def __init__(
        self,
        config: WorkflowConfig,
        csv_path: Optional[str] = None,
        operators_json_path: str = "operators.json",
    ):
        self.config = config
        self.csv_path = csv_path or self.resolve_field_catalog_path()
        self.operators_json_path = operators_json_path

    def resolve_field_catalog_path(self) -> Optional[str]:
        if self.config.field_catalog_path:
            resolved = resolve_field_catalog_path(self.config.field_catalog_path)
            return str(resolved) if resolved else None

        normalized = self.normalized_settings()
        field_catalog_dir = Path(__file__).parent / "field_catalogs"
        candidates = [
            field_catalog_dir / f"{normalized['region']}_{normalized['universe']}_D{normalized['delay']}.csv",
            field_catalog_dir / f"{normalized['region']}_{normalized['universe']}_{normalized['delay']}.csv",
            field_catalog_dir / f"{normalized['region']}_{normalized['universe']}.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        return None

    def field_catalog_status(self) -> Dict[str, Any]:
        resolved = resolve_field_catalog_path(self.csv_path) if self.csv_path else None
        field_count = 0
        if resolved and resolved.is_file():
            try:
                lines = [line.strip() for line in resolved.read_text(encoding="utf-8", errors="ignore").splitlines()]
                non_empty_lines = [line for line in lines if line]
                field_count = max(len(non_empty_lines) - 1, 0)
            except Exception:
                field_count = 0
        return {
            "field_catalog_path": str(resolved) if resolved else None,
            "field_catalog_loaded": bool(resolved and resolved.is_file() and field_count > 0),
            "field_catalog_degraded": not bool(resolved and resolved.is_file() and field_count > 0),
            "field_count": field_count,
        }

    def normalized_settings(self) -> Dict[str, Any]:
        region = REGION_ALIAS_MAP.get(self.config.region.upper(), self.config.region.upper())
        return {
            "instrument_type": self.config.instrument_type,
            "region": region,
            "universe": self.config.universe,
            "delay": self.config.delay,
            "neutralization": self.config.neutralization,
            "language": self.config.language,
        }

    def build_research_brief(self) -> Dict[str, Any]:
        return {
            "region_requested": self.config.region,
            "region_normalized": self.normalized_settings()["region"],
            "field_types": self.config.field_types,
            "baseline_alpha_id": self.config.baseline_alpha_id,
            "field_catalog": self.field_catalog_status(),
            "research_policy": asdict(self.config.research),
            "search_objective": {
                "priority": "Authoritative finance and quant literature first",
                "minimum_sources": self.config.research.min_paper_count,
                "required_outputs": [
                    "factor intuition summary",
                    "economic mechanism statement",
                    "field mapping candidates",
                    "robustness caveats",
                    "region-specific implementation risks",
                    "pyramid contribution plan",
                    "consultant income relevance",
                ],
            },
            "income_optimization_bias": {
                "prioritize_pyramids": self.config.automation.prioritize_pyramids,
                "prioritize_income_multipliers": self.config.automation.prioritize_income_multipliers,
                "stage_scope": self.config.automation.stage_scope,
                "region_allocation": dict(self.config.automation.region_allocation),
                "priority_targets": list(self.config.automation.pyramid_priority_targets),
                "priority_table": self.current_quarter_priority_table(),
                "target": "Expand diversified consultant pyramids while preserving deployable alpha quality.",
            },
        }

    def build_automation_brief(self) -> Dict[str, Any]:
        return {
            "single_round_goal_required": self.config.automation.single_round_goal_required,
            "frozen_directions": list(self.config.automation.frozen_directions),
            "active_family": self.config.automation.active_family,
            "research_memory_path": self.config.automation.research_memory_path,
            "experience_library_path": self.config.automation.experience_library_path,
            "experience_library_summary": self.load_experience_library_summary(),
            "economic_rationale_gate": self.config.research.require_economic_rationale_before_expression,
            "research_refresh_sources": self.default_research_refresh_sources(),
            "daily_qualified_alpha_goal": self.config.automation.min_qualified_alphas_per_day,
            "stage_scope": self.config.automation.stage_scope,
            "region_allocation": dict(self.config.automation.region_allocation),
            "pyramid_priority_targets": list(self.config.automation.pyramid_priority_targets),
            "pyramid_priority_table": self.current_quarter_priority_table(),
            "batch_region_mix": self.compute_batch_region_mix(),
            "planner_priorities": [
                "Do not write a new expression until the intended economic mechanism is stated clearly.",
                "Classify failure type before generating the next batch.",
                "Keep one round focused on one primary goal.",
                "Prefer the smallest repair action that matches the failure type.",
                "Mark plateaued families explicitly and stop expanding them as the mainline.",
                "When plateau appears, search literature, forum posts, and stored mining lessons before launching another local retry batch.",
                "Use external research to widen the hypothesis set instead of repeatedly touching the same bottleneck knobs.",
                "Bias exploration toward new pyramid-forming dataset families and regions when quality is comparable.",
                "Do not count a day as successful unless at least two alphas meet consultant-grade floors.",
                "Keep the current quarter pyramid plan explicit: EUR expansion first, MEA second, ASI third.",
                "Use ASI as a secondary expansion lane rather than the only mainline until daily qualified output is stable.",
            ],
        }

    def current_quarter_priority_table(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for raw in self.config.automation.pyramid_priority_table:
            if not isinstance(raw, dict):
                continue
            parsed = _parse_pyramid_target(raw.get("target"))
            if not parsed:
                continue
            row = dict(raw)
            row.update(parsed)
            rows.append(row)
        rows.sort(key=lambda row: row.get("priority", 1e9))
        return rows

    def compute_batch_region_mix(self) -> Dict[str, int]:
        allocation = dict(self.config.automation.region_allocation)
        normalized_total = sum(weight for weight in allocation.values() if isinstance(weight, (int, float)) and weight > 0)
        if normalized_total <= 0:
            allocation = {"EUR": 0.50, "MEA": 0.25, "ASI": 0.25}
            normalized_total = 1.0
        raw_counts = {
            region: (weight / normalized_total) * self.config.batch_size
            for region, weight in allocation.items()
            if isinstance(weight, (int, float)) and weight > 0
        }
        floor_counts = {region: int(raw_counts[region]) for region in raw_counts}
        assigned = sum(floor_counts.values())
        remainder_order = sorted(
            raw_counts.keys(),
            key=lambda region: (raw_counts[region] - floor_counts[region], raw_counts[region]),
            reverse=True,
        )
        remainder_idx = 0
        while assigned < self.config.batch_size and remainder_order:
            region = remainder_order[remainder_idx % len(remainder_order)]
            floor_counts[region] += 1
            assigned += 1
            remainder_idx += 1
        return floor_counts

    def recommended_priority_targets(self, region: Optional[str] = None, limit: int = 6) -> List[Dict[str, Any]]:
        rows = self.current_quarter_priority_table()
        if region:
            rows = [row for row in rows if row.get("region") == region]
        return rows[:limit]

    def load_experience_library(self) -> Optional[Dict[str, Any]]:
        path_value = self.config.automation.experience_library_path
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_file():
            return None
        return _safe_json_load(path)

    def load_experience_library_summary(self) -> Dict[str, Any]:
        payload = self.load_experience_library()
        if not isinstance(payload, dict):
            return {}
        families = payload.get("family_status", {}) or {}
        keep = [name for name, info in families.items() if isinstance(info, dict) and info.get("status") == "keep"]
        reference = [name for name, info in families.items() if isinstance(info, dict) and info.get("status") == "reference"]
        freeze = [name for name, info in families.items() if isinstance(info, dict) and info.get("status") == "freeze"]
        return {
            "scope": payload.get("scope"),
            "keep_families": keep,
            "reference_families": reference,
            "freeze_families": freeze,
        }

    def resolve_active_family_experience(self) -> Optional[Dict[str, Any]]:
        payload = self.load_experience_library()
        if not isinstance(payload, dict):
            return None
        family_status = payload.get("family_status", {}) or {}
        active_family = (self.config.automation.active_family or "").strip().lower()
        if not active_family:
            return None

        for name, info in family_status.items():
            if str(name).lower() == active_family:
                row = dict(info) if isinstance(info, dict) else {"status": str(info)}
                row["family_name"] = name
                return row

        alias_map = {
            "analyst11": "analyst11_esg_percentile_delta",
            "analyst69": "analyst69_recommendation_count_breadth",
            "analyst44": "analyst44_broker_revision_vectors",
            "analyst15": "analyst15_disagreement_revision_breadth",
            "analyst14": "analyst14_recommendation_breadth",
            "analyst10": "analyst10_predicted_surprise",
            "analyst4": "analyst4_recommendation_vector",
        }
        for token, name in alias_map.items():
            if token in active_family and name in family_status:
                info = family_status.get(name) or {}
                row = dict(info) if isinstance(info, dict) else {"status": str(info)}
                row["family_name"] = name
                return row
        return None

    def default_research_refresh_sources(self) -> List[str]:
        return [
            "literature",
            "forum_posts",
            "experience_library",
        ]

    def build_research_refresh_requirement(
        self,
        *,
        failure_type: Optional[str],
        plateau_info: Optional[Dict[str, Any]],
        family_status: Optional[str],
    ) -> Dict[str, Any]:
        plateau_detected = bool((plateau_info or {}).get("plateau_detected"))
        require_refresh = plateau_detected or family_status in {"freeze", "reference"}
        if not require_refresh:
            return {
                "required": False,
                "reason": None,
                "sources": [],
                "required_outputs": [],
            }

        reason = "Plateau detected; new research input is required before another same-line retry."
        if family_status == "freeze":
            reason = "Active family is frozen by the experience library; switch thesis after refreshing research."
        elif family_status == "reference":
            reason = "Active family is reference-only; research refresh is required before choosing the next mainline."
        elif failure_type == "temporal_stability_failure":
            reason = "Stability branch is plateaued; refresh the hypothesis set before more local variants."

        return {
            "required": True,
            "reason": reason,
            "sources": self.default_research_refresh_sources(),
            "required_outputs": [
                "economic mechanism statement",
                "field-to-mechanism mapping",
                "reason this branch should reduce the current blocker",
                "how the new idea expands current-quarter pyramid coverage",
            ],
        }

    def build_integrated_workflow_heuristics(self) -> Dict[str, Any]:
        heuristics = WorkflowHeuristics(
            platform_detection_rules=[
                "If neighboring parameter variants produce nearly identical Sharpe/Fitness/correlation while the same hard check still fails, treat the line as a local plateau.",
                "If tightening a regime threshold improves one metric but repeatedly damages sub-universe or correlation, stop treating the threshold as the primary lever.",
                "When a branch has passed all checks except one stubborn stability metric, prefer changing branch shape before changing core fields.",
                "When a plateau is detected, require an external idea refresh from literature or peer experience before approving another same-family retry.",
            ],
            metric_resolution_order=[
                "First make the expression valid locally and keep platform operatorCount within cap.",
                "Then stabilize the core floor metrics: Sharpe, Fitness, Turnover, Concentration, Sub-universe.",
                "Only after the floor metrics are stable should optimization focus on ladder and production correlation.",
                "When only one hard metric remains, isolate the branch most likely responsible and modify that branch instead of rewriting the full expression.",
            ],
            late_stage_priorities=[
                "Do not keep chasing higher Sharpe once the main floor metrics already pass.",
                "Prefer branch-shape substitution over global field replacement.",
                "Use same-field alternative time-shape transforms such as rank vs zscore vs scale when ladder is the final blocker.",
                "If correlation is near the gate, prioritize local decorrelation in the smallest possible branch.",
            ],
            final_breakthrough_patterns=[
                "A frozen low-regime branch can stay intact while the else branch is replaced.",
                "Local auxiliary-field substitution can break a production-correlation plateau without destroying the main signal.",
                "Time-shape substitution may unlock ladder pass where parameter tuning cannot.",
            ],
        )
        return asdict(heuristics)

    def determine_stage(
        self,
        best_sharpe: Optional[float],
        best_fitness: Optional[float],
    ) -> str:
        if best_sharpe is not None and best_fitness is not None:
            if best_sharpe > 1.40 and best_fitness > 0.90:
                return STAGE_B
        return STAGE_A

    def build_batch_blueprint(self, stage: str) -> List[CandidateBlueprint]:
        slots: List[CandidateBlueprint] = []
        themes = [
            ["A", "F"],
            ["B", "C"],
            ["D", "E"],
            ["A", "D"],
            ["B", "F"],
            ["C", "E"],
            ["A", "E"],
            ["D", "F"],
        ]
        region_targets: Dict[str, List[str]] = {}
        for target in self.config.automation.pyramid_priority_targets:
            parts = [part.strip() for part in target.split(":")]
            if len(parts) != 3:
                continue
            region_targets.setdefault(parts[0], []).append(f"{parts[1]}:{parts[2]}")

        allocation = dict(self.config.automation.region_allocation)
        normalized_total = sum(weight for weight in allocation.values() if isinstance(weight, (int, float)) and weight > 0)
        if normalized_total <= 0:
            allocation = {"EUR": 0.50, "MEA": 0.25, "ASI": 0.25}
            normalized_total = 1.0

        raw_counts: Dict[str, float] = {
            region: (weight / normalized_total) * self.config.batch_size
            for region, weight in allocation.items()
            if isinstance(weight, (int, float)) and weight > 0
        }
        floor_counts: Dict[str, int] = {region: int(raw_counts[region]) for region in raw_counts}
        assigned = sum(floor_counts.values())
        remainder_order = sorted(
            raw_counts.keys(),
            key=lambda region: (raw_counts[region] - floor_counts[region], raw_counts[region]),
            reverse=True,
        )
        remainder_idx = 0
        while assigned < self.config.batch_size and remainder_order:
            region = remainder_order[remainder_idx % len(remainder_order)]
            floor_counts[region] += 1
            assigned += 1
            remainder_idx += 1

        ordered_regions = [region for region in allocation.keys() if region in floor_counts]
        remaining_regions = [region for region in floor_counts.keys() if region not in ordered_regions]
        slot_regions: List[str] = []
        for region in ordered_regions + remaining_regions:
            slot_regions.extend([region] * floor_counts[region])
        if len(slot_regions) < self.config.batch_size:
            slot_regions.extend(["EUR"] * (self.config.batch_size - len(slot_regions)))
        slot_regions = slot_regions[: self.config.batch_size]

        for idx in range(self.config.batch_size):
            if stage == STAGE_B and idx < 5:
                role = "Exploit"
                family = "parameter-refine or controlled structure extension"
            else:
                role = "Explore"
                family = "structure-driven diversification"
            focus_region = slot_regions[idx]
            priority_pyramids = region_targets.get(focus_region, [])
            slots.append(
                CandidateBlueprint(
                    slot=idx + 1,
                    role=role,
                    strategy_family=family,
                    focus_region=focus_region,
                    required_themes=themes[idx],
                    priority_pyramids=priority_pyramids[:3],
                    notes=[
                        "State the economic mechanism before proposing the expression.",
                        "Use frozen core fields.",
                        "Avoid pure fine-tune unless stage gate allows it.",
                        "Keep estimated operator count within cap.",
                        "Bias new slots toward diversified datasets, operators, and pyramid expansion.",
                        "Prefer consultant-income-positive shapes: low correlation, stable ladder, investable turnover.",
                        f"Primary region allocation for this slot is {focus_region}.",
                        f"Prefer these pyramid gaps first: {', '.join(priority_pyramids[:3]) if priority_pyramids else 'follow current-quarter priority gaps'}.",
                    ],
                )
            )
        return slots

    def preflight_expression(self, candidate: Any) -> CandidatePreflight:
        normalized_candidate = _normalize_expression_candidate(candidate)
        expression = normalized_candidate["expression"]
        economic_rationale = normalized_candidate["economic_rationale"]
        fingerprint = asdict(fingerprint_expression(expression))
        validation = validate_expression_detailed(
            expression=expression,
            csv_path=self.csv_path,
            operators_json_path=self.operators_json_path,
        )
        issues: List[str] = []

        if not expression:
            issues.append("Expression is empty.")
        if fingerprint["operator_count_estimate"] > self.config.max_operator_count:
            issues.append(
                f"Estimated operator count {fingerprint['operator_count_estimate']} exceeds cap {self.config.max_operator_count}."
            )
        if self.config.research.require_economic_rationale_before_expression:
            rationale_text = (economic_rationale or "").strip()
            rationale_word_count = len([token for token in rationale_text.replace("/", " ").split() if token])
            if not rationale_text:
                issues.append(
                    "Missing economic rationale: each expression must state the intended return mechanism before it can enter serious mining."
                )
            elif len(rationale_text) < 24 or rationale_word_count < 4:
                issues.append(
                    "Economic rationale is too thin: explain the field behavior, why mispricing should exist, and why the shape should monetize it."
                )
        if len(fingerprint["common_operator_hits"]) >= 2 and not fingerprint["theme_operators"]:
            issues.append("Expression uses multiple common operators without any A-F theme operator.")
        if not validation["is_valid"]:
            issues.extend(validation["errors"])
        else:
            risky_addition = (
                "add(" in expression
                and "rank(" not in expression
                and "zscore(" not in expression
                and "scale(" not in expression
                and "group_rank(" not in expression
                and "group_scale(" not in expression
            )
            if risky_addition:
                issues.append(
                    "Raw add-shape detected without an outer cross-sectional normalization. This pattern has repeatedly produced platform-side unit or concentration problems; prefer rank/zscore/scale/group transforms unless the unit story is explicit."
                )
        if validation.get("field_catalog", {}).get("field_catalog_degraded"):
            issues.append("Field catalog degraded: field-name/type checks are running in reduced-confidence mode.")

        return CandidatePreflight(
            expression=expression,
            economic_rationale=economic_rationale,
            fingerprint=fingerprint,
            validation=validation,
            issues=issues,
        )

    def preflight_batch(self, expressions: Sequence[Any]) -> Dict[str, Any]:
        normalized_candidates = [_normalize_expression_candidate(expr) for expr in expressions]
        reports = [self.preflight_expression(expr) for expr in expressions]
        theme_coverage = sorted(
            {
                theme
                for report in reports
                for theme in report.fingerprint["themes"]
            }
        )
        common_usage: Dict[str, int] = {}
        for report in reports:
            for operator in report.fingerprint["common_operator_hits"]:
                common_usage[operator] = common_usage.get(operator, 0) + 1

        batch_issues: List[str] = []
        if len(expressions) != self.config.batch_size:
            batch_issues.append(
                f"Batch must contain exactly {self.config.batch_size} expressions, got {len(expressions)}."
            )
        if len(theme_coverage) < 4:
            batch_issues.append(
                f"Batch theme coverage is {theme_coverage}; at least 4 A-F themes are required."
            )
        for operator, count in sorted(common_usage.items()):
            if count > 2:
                batch_issues.append(f"Common operator '{operator}' appears {count} times; cap is 2.")
        if self.config.research.require_economic_rationale_before_expression:
            missing_rationale_count = sum(1 for item in normalized_candidates if not item.get("economic_rationale"))
            if missing_rationale_count:
                batch_issues.append(
                    f"{missing_rationale_count} candidate(s) are missing an economic rationale. Structured rationale is required before simulation."
                )

        return {
            "batch_size": len(expressions),
            "theme_coverage": theme_coverage,
            "common_operator_usage": common_usage,
            "batch_issues": batch_issues,
            "stage_gate_note": (
                "Stage A forbids pure fine-tuning until a current best alpha exceeds Sharpe 1.40 and Fitness 0.90."
            ),
            "candidates": [
                {
                    "expression": report.expression,
                    "economic_rationale": report.economic_rationale,
                    "fingerprint": report.fingerprint,
                    "validation": report.validation,
                    "issues": report.issues,
                }
                for report in reports
            ],
        }

    def diagnose_alpha_summary(self, alpha_summary: Dict[str, Any]) -> Dict[str, Any]:
        checks = alpha_summary.get("checks", {}) or {}
        diagnoses: List[MetricDiagnosis] = []

        def add(metric: str, status: str, observed_value: Any, diagnosis: str, actions: List[str]) -> None:
            diagnoses.append(
                MetricDiagnosis(
                    metric=metric,
                    status=status,
                    observed_value=observed_value,
                    diagnosis=diagnosis,
                    recommended_actions=actions,
                )
            )

        sharpe = alpha_summary.get("sharpe")
        fitness = alpha_summary.get("fitness")
        turnover = alpha_summary.get("turnover")
        prod_corr = ((checks.get("PROD_CORRELATION", {}) or {}).get("value"))
        self_corr = ((checks.get("SELF_CORRELATION", {}) or {}).get("value"))
        stability_name = _stability_check_name(alpha_summary)
        stability_check = _stability_check(alpha_summary)
        ladder = stability_check.get("value")
        sub_universe = ((checks.get("LOW_SUB_UNIVERSE_SHARPE", {}) or {}).get("value"))

        if isinstance(sharpe, (int, float)) and sharpe < self.config.targets.min_sharpe:
            add(
                "SHARPE",
                "FAIL",
                sharpe,
                "Signal strength is below the workflow floor. This is usually a structure problem, not a parameter problem.",
                [
                    "Recheck whether the field combination carries a coherent financial story.",
                    "Prefer structural changes before any fine-tune.",
                    "Check whether smoothing or branch logic is flattening the signal.",
                ],
            )
        if isinstance(fitness, (int, float)) and fitness < self.config.targets.min_fitness:
            add(
                "FITNESS",
                "FAIL",
                fitness,
                "Risk-adjusted efficiency is weak. The signal may be noisy or too unstable across the sample.",
                [
                    "Check whether turnover or drawdown is diluting a decent Sharpe.",
                    "Try structure changes that improve stability before increasing raw aggressiveness.",
                    "Avoid adding fields without a clear role in the expression.",
                ],
            )
        if isinstance(sub_universe, (int, float)) and checks.get("LOW_SUB_UNIVERSE_SHARPE", {}).get("result") == "FAIL":
            add(
                "SUB_UNIVERSE",
                "FAIL",
                sub_universe,
                "Robustness across a stricter universe is failing. The current signal is likely too regime-specific or too narrow.",
                [
                    "Loosen brittle regime thresholds before adding complexity.",
                    "Prefer group or scale adjustments that smooth the cross-section.",
                    "Do not trust headline Sharpe while sub-universe still fails.",
                ],
            )
        if isinstance(self_corr, (int, float)) and self_corr >= self.config.targets.max_self_correlation:
            add(
                "SELF_CORRELATION",
                "FAIL",
                self_corr,
                "Self-correlation is too high. This branch is too close to your own pool and weakens pool diversity and Osmosis value.",
                [
                    "Prefer sibling diversification over local parameter churn.",
                    "Swap confirmation fields or operator shape inside the smallest branch.",
                    "Preserve the core intuition but make the expression family less redundant.",
                ],
            )
        if checks.get("CONCENTRATED_WEIGHT", {}).get("result") == "FAIL":
            add(
                "CONCENTRATION",
                "FAIL",
                None,
                "Portfolio weights are too sharp. The cross-sectional transformation is over-concentrating exposure.",
                [
                    "Try scale-style or group-style transforms before introducing more fields.",
                    "Avoid piling on additional operators that raise operatorCount without fixing shape.",
                    "Treat concentration failures as structural invalidation, not a cosmetic issue.",
                ],
            )
        if isinstance(ladder, (int, float)) and stability_check.get("result") != "PASS":
            add(
                stability_name,
                "FAIL",
                ladder,
                "Cross-period stability is still insufficient. This is often a branch-shape problem rather than a simple parameter problem.",
                [
                    "Check whether the current branch is trapped in a local plateau.",
                    "Prefer same-field shape substitution such as rank vs zscore vs scale before changing core fields.",
                    "If only ladder remains, modify the most responsible branch instead of rewriting the full expression.",
                ],
            )
        if isinstance(prod_corr, (int, float)) and prod_corr >= self.config.targets.max_prod_correlation:
            add(
                "PROD_CORRELATION",
                "FAIL",
                prod_corr,
                "Production correlation is too high. The expression is too close to existing production structure.",
                [
                    "Prefer local decorrelation in one branch over global rewrites.",
                    "Substitute auxiliary fields inside the most correlated branch before changing the whole alpha.",
                    "Do not assume small parameter changes will materially reduce correlation once the line is pinned.",
                ],
            )
        if not diagnoses:
            add(
                "ALL_CORE_CHECKS",
                "PASS",
                None,
                "All current core checks pass under workflow floors. Remaining work, if any, should focus on target-band improvements rather than rescue actions.",
                [
                    "Protect the current structure and avoid unnecessary rewrites.",
                    "Use submission checks to verify production correlation before declaring done.",
                ],
            )

        return {
            "alpha_id": alpha_summary.get("alpha_id") or alpha_summary.get("id"),
            "diagnoses": [asdict(item) for item in diagnoses],
        }

    def classify_failure_type(self, alpha_summary: Dict[str, Any]) -> Dict[str, Any]:
        checks = alpha_summary.get("checks", {}) or {}
        sharpe = alpha_summary.get("sharpe")
        fitness = alpha_summary.get("fitness")
        prod_corr = (checks.get("PROD_CORRELATION", {}) or {}).get("value")
        self_corr = (checks.get("SELF_CORRELATION", {}) or {}).get("value")

        sharpe_fail = isinstance(sharpe, (int, float)) and sharpe < self.config.targets.min_sharpe
        fitness_fail = isinstance(fitness, (int, float)) and fitness < self.config.targets.min_fitness
        subu_fail = checks.get("LOW_SUB_UNIVERSE_SHARPE", {}).get("result") == "FAIL"
        concentration_fail = checks.get("CONCENTRATED_WEIGHT", {}).get("result") == "FAIL"
        ladder_fail = not _stability_is_passing(alpha_summary)
        prod_corr_fail = isinstance(prod_corr, (int, float)) and prod_corr >= self.config.targets.max_prod_correlation
        self_corr_fail = isinstance(self_corr, (int, float)) and self_corr >= self.config.targets.max_self_correlation

        if sharpe_fail and fitness_fail:
            return {
                "failure_type": "structure_failure",
                "reason": "Sharpe and fitness are both below workflow floors.",
                "primary_goal": "structure_rebuild",
            }
        if concentration_fail:
            return {
                "failure_type": "concentration_failure",
                "reason": "Concentration failed; cross-sectional shape is too sharp.",
                "primary_goal": "concentration_repair",
            }
        if subu_fail:
            return {
                "failure_type": "breadth_failure",
                "reason": "Sub-universe failed while the main signal may still be viable.",
                "primary_goal": "sub_universe_repair",
            }
        if ladder_fail:
            return {
                "failure_type": "temporal_stability_failure",
                "reason": "Cross-period stability remains the blocker.",
                "primary_goal": "ladder_repair",
            }
        if prod_corr_fail:
            return {
                "failure_type": "correlation_failure",
                "reason": "Production correlation is above the workflow cap.",
                "primary_goal": "correlation_push",
            }
        if self_corr_fail:
            return {
                "failure_type": "correlation_failure",
                "reason": "Self correlation is above the workflow cap.",
                "primary_goal": "correlation_push",
            }
        if sharpe_fail or fitness_fail:
            return {
                "failure_type": "setting_failure",
                "reason": "Only one core efficiency metric is below floor and structure may still be usable.",
                "primary_goal": "fitness_push" if fitness_fail else "sharpe_push",
            }
        return {
            "failure_type": "passed_floor",
            "reason": "All current floor checks pass.",
            "primary_goal": "target_band_refine",
        }

    def action_library_for_failure_type(self, failure_type: str) -> Dict[str, List[str]]:
        libraries = {
            "structure_failure": {
                "allowed": [
                    "replace core field family",
                    "switch expression skeleton",
                    "move to a new dataset family",
                ],
                "blocked": [
                    "settings-only sweep",
                    "cosmetic outer wrapper changes",
                ],
            },
            "breadth_failure": {
                "allowed": [
                    "same-dataset coverage modulation",
                    "analyst-count or breadth modulation",
                    "small neutralization comparison",
                ],
                "blocked": [
                    "full signal rewrite",
                ],
            },
            "concentration_failure": {
                "allowed": [
                    "outer compression",
                    "group_scale or group_rank reshaping",
                    "truncation neighborhood test",
                    "light breadth dilution",
                ],
                "blocked": [
                    "adding many new fields at once",
                ],
            },
            "temporal_stability_failure": {
                "allowed": [
                    "time-shape substitution",
                    "branch-local temporal rewrite",
                    "new sibling family search",
                ],
                "blocked": [
                    "long setting-only grind",
                ],
            },
            "correlation_failure": {
                "allowed": [
                    "open sibling decorrelation branch",
                    "swap auxiliary confirmation field",
                    "local branch decorrelation",
                ],
                "blocked": [
                    "rewrite the current best line globally",
                ],
            },
            "setting_failure": {
                "allowed": [
                    "small decay sweep",
                    "small truncation sweep",
                    "industry versus subindustry comparison",
                ],
                "blocked": [
                    "large structural expansion",
                ],
            },
            "passed_floor": {
                "allowed": [
                    "protect winner",
                    "run submission checks",
                    "target-band refinement only",
                ],
                "blocked": [
                    "unnecessary rebuild",
                ],
            },
        }
        return libraries.get(failure_type, {"allowed": [], "blocked": []})

    def detect_plateau(self, alpha_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if len(alpha_summaries) < self.config.automation.plateau_min_neighbor_count:
            return {"plateau_detected": False, "reason": None}

        ladders = [_stability_check(summary).get("value") for summary in alpha_summaries]
        valid_ladders = [value for value in ladders if isinstance(value, (int, float))]
        if len(valid_ladders) >= self.config.automation.plateau_min_neighbor_count:
            spread = max(valid_ladders) - min(valid_ladders)
            if spread <= self.config.automation.plateau_metric_spread_threshold:
                metric_name = _stability_check_name(alpha_summaries[0])
                return {
                    "plateau_detected": True,
                    "reason": "Neighboring ladder results are tightly clustered, indicating a local plateau.",
                    "metric": metric_name,
                    "spread": spread,
                }

        fitness_values = [
            summary.get("fitness")
            for summary in alpha_summaries
            if isinstance(summary.get("fitness"), (int, float))
        ]
        if len(fitness_values) >= self.config.automation.plateau_min_neighbor_count:
            spread = max(fitness_values) - min(fitness_values)
            if spread <= self.config.automation.plateau_improvement_threshold:
                return {
                    "plateau_detected": True,
                    "reason": "Neighboring fitness values are nearly identical, indicating low marginal return from local tweaks.",
                    "metric": "FITNESS",
                    "spread": spread,
                }

        return {"plateau_detected": False, "reason": None}

    def plan_next_round(
        self,
        best_summary: Optional[Dict[str, Any]],
        batch_failure_counts: Dict[str, int],
        plateau_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        stage = self.determine_stage(
            best_summary.get("sharpe") if best_summary else None,
            best_summary.get("fitness") if best_summary else None,
        )
        next_batch_blueprint = [asdict(item) for item in self.build_batch_blueprint(stage)]
        batch_region_mix = self.compute_batch_region_mix()
        priority_targets = self.recommended_priority_targets(limit=6)
        current_region_targets = self.recommended_priority_targets(self.normalized_settings()["region"], limit=4)
        family_experience = self.resolve_active_family_experience()

        if best_summary is None:
            research_refresh_requirement = self.build_research_refresh_requirement(
                failure_type="structure_failure",
                plateau_info=plateau_info,
                family_status=family_experience.get("status") if isinstance(family_experience, dict) else None,
            )
            plan = {
                "primary_goal": "structure_rebuild",
                "failure_type": "structure_failure",
                "reason": "No usable alpha summary was available.",
                "allowed_actions": self.action_library_for_failure_type("structure_failure")["allowed"],
                "blocked_actions": self.action_library_for_failure_type("structure_failure")["blocked"],
                "family_status": "active",
                "research_refresh_required": research_refresh_requirement["required"],
                "research_refresh": research_refresh_requirement,
                "stage": stage,
                "stage_scope": self.config.automation.stage_scope,
                "batch_region_mix": batch_region_mix,
                "recommended_priority_targets": priority_targets,
                "current_region_priority_targets": current_region_targets,
                "next_batch_blueprint": next_batch_blueprint,
            }
            return self.apply_experience_constraints_to_plan(plan, family_experience)

        classification = self.classify_failure_type(best_summary)
        failure_type = classification["failure_type"]
        action_library = self.action_library_for_failure_type(failure_type)
        family_status = "plateaued" if plateau_info.get("plateau_detected") else "active"
        family_experience_status = family_experience.get("status") if isinstance(family_experience, dict) else None

        if plateau_info.get("plateau_detected") and failure_type == "temporal_stability_failure":
            classification = {
                "failure_type": "temporal_stability_failure",
                "reason": plateau_info.get("reason"),
                "primary_goal": "new_family_validation",
            }
            action_library = {
                "allowed": [
                    "open a new sibling family",
                    "switch to a new dataset family",
                    "preserve the current line as control",
                    "review literature and peer mining examples before selecting the next sibling",
                ],
                "blocked": [
                    "continue expanding the current mainline",
                ],
            }

        research_refresh_requirement = self.build_research_refresh_requirement(
            failure_type=classification["failure_type"],
            plateau_info=plateau_info,
            family_status=family_experience_status,
        )
        plan = {
            "primary_goal": classification["primary_goal"],
            "failure_type": classification["failure_type"],
            "reason": classification["reason"],
            "allowed_actions": action_library["allowed"],
            "blocked_actions": action_library["blocked"],
            "single_goal_enforced": self.config.automation.single_round_goal_required,
            "family_status": family_status,
            "research_refresh_required": research_refresh_requirement["required"],
            "research_refresh": research_refresh_requirement,
            "frozen_directions": list(self.config.automation.frozen_directions),
            "active_family": self.config.automation.active_family,
            "batch_failure_counts": batch_failure_counts,
            "stage": stage,
            "stage_scope": self.config.automation.stage_scope,
            "batch_region_mix": batch_region_mix,
            "recommended_priority_targets": priority_targets,
            "current_region_priority_targets": current_region_targets,
            "next_batch_blueprint": next_batch_blueprint,
        }
        return self.apply_experience_constraints_to_plan(plan, family_experience)

    def apply_experience_constraints_to_plan(
        self,
        plan: Dict[str, Any],
        family_experience: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(family_experience, dict):
            return plan

        status = family_experience.get("status")
        family_name = family_experience.get("family_name")
        plan["experience_family"] = family_name
        plan["experience_status"] = status
        plan["experience_reason"] = family_experience.get("reason")

        if status == "freeze":
            plan["family_status"] = "frozen_by_library"
            plan["primary_goal"] = "new_family_validation"
            plan["reason"] = f"Active family is frozen in the experience library. {family_experience.get('reason') or ''}".strip()
            plan["allowed_actions"] = [
                "switch to a new family from the current priority table",
                "preserve frozen family only as negative reference",
                "open a new structure in the highest-priority region/category gap",
            ]
            plan["blocked_actions"] = sorted(set(list(plan.get("blocked_actions", [])) + [
                "continue current frozen family as mainline",
                "settings-only sweep on frozen family",
            ]))
        elif status == "reference":
            plan["family_status"] = "reference_only"
            plan["reason"] = f"Active family is retained as historical reference, not as the preferred mainline. {family_experience.get('reason') or ''}".strip()
            plan["allowed_actions"] = sorted(set(list(plan.get("allowed_actions", [])) + [
                "use active family only as control/reference",
                "compare against a new family from the priority table",
            ]))
            plan["blocked_actions"] = sorted(set(list(plan.get("blocked_actions", [])) + [
                "continue current reference family as sole mainline",
            ]))
        elif status == "keep":
            plan["allowed_actions"] = sorted(set(list(plan.get("allowed_actions", [])) + [
                "protect known valid family structure",
            ]))
        return plan

    def infer_memory_outcome(
        self,
        best_candidate: Optional[Dict[str, Any]],
        next_round_plan: Dict[str, Any],
        batch_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not best_candidate:
            return {
                "status": "no_result",
                "summary": "No matched candidate was available for memory classification.",
            }

        stop_condition = (best_candidate.get("stop_condition") or {})
        if stop_condition.get("hits_target_band"):
            return {
                "status": "target_hit",
                "summary": "The leading candidate hit the workflow target band.",
            }
        if stop_condition.get("meets_floor"):
            return {
                "status": "floor_pass",
                "summary": "The leading candidate passed workflow floors but did not hit the target band.",
            }
        if next_round_plan.get("family_status") == "plateaued":
            return {
                "status": "plateaued",
                "summary": batch_analysis.get("plateau_reason") or "The family is locally plateaued.",
            }
        if next_round_plan.get("failure_type") == "structure_failure":
            return {
                "status": "structure_fail",
                "summary": "The batch still fails on multiple core metrics and needs a rebuild.",
            }
        return {
            "status": "partial_progress",
            "summary": "The batch preserved a viable line but still needs focused repair.",
        }

    def build_research_memory_entry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        best_candidate = (payload.get("best_candidate") or {})
        next_round_plan = (payload.get("next_round_plan") or {})
        batch_analysis = (payload.get("batch_analysis") or {})
        result_summary = self.infer_memory_outcome(best_candidate, next_round_plan, batch_analysis)

        return {
            "round": payload.get("round"),
            "region": payload.get("region"),
            "universe": payload.get("universe"),
            "delay": payload.get("delay"),
            "neutralization": payload.get("neutralization"),
            "stage_scope": ((payload.get("automation_context") or {}).get("stage_scope")),
            "batch_region_mix": ((payload.get("next_round_plan") or {}).get("batch_region_mix")) or {},
            "recommended_priority_targets": ((payload.get("next_round_plan") or {}).get("recommended_priority_targets")) or [],
            "active_family": ((payload.get("automation_context") or {}).get("active_family")),
            "frozen_directions": ((payload.get("automation_context") or {}).get("frozen_directions")) or [],
            "best_alpha_id": best_candidate.get("alpha_id"),
            "best_metrics": {
                "sharpe": _safe_float(best_candidate.get("sharpe")),
                "fitness": _safe_float(best_candidate.get("fitness")),
                "turnover": _safe_float(best_candidate.get("turnover")),
                "prod_corr": _safe_float(best_candidate.get("prod_corr")),
                "self_corr": _safe_float(best_candidate.get("self_corr")),
                "ladder_value": _safe_float(best_candidate.get("ladder_value")),
            },
            "failure_type": next_round_plan.get("failure_type"),
            "primary_goal": next_round_plan.get("primary_goal"),
            "family_status": next_round_plan.get("family_status"),
            "allowed_actions": next_round_plan.get("allowed_actions", []),
            "blocked_actions": next_round_plan.get("blocked_actions", []),
            "outcome_status": result_summary["status"],
            "outcome_summary": result_summary["summary"],
            "batch_failure_counts": batch_analysis.get("batch_failure_counts", {}),
            "plateau_info": batch_analysis.get("plateau_info"),
            "notes": payload.get("notes", []),
        }

    def summarize_research_memory(self, log_path: str) -> Dict[str, Any]:
        path = Path(log_path)
        if not path.is_file():
            return {
                "log_path": log_path,
                "entries": 0,
                "failure_type_summary": {},
                "action_effectiveness": {},
                "family_status_summary": {},
                "outcome_status_summary": {},
            }

        failure_type_summary: Dict[str, int] = {}
        family_status_summary: Dict[str, int] = {}
        outcome_status_summary: Dict[str, int] = {}
        action_effectiveness: Dict[str, Dict[str, Dict[str, int]]] = {}
        entries = 0

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            payload = line.get("payload", {}) or {}
            memory = payload.get("research_memory")
            if not isinstance(memory, dict):
                continue
            entries += 1

            failure_type = memory.get("failure_type") or "unknown"
            family_status = memory.get("family_status") or "unknown"
            outcome_status = memory.get("outcome_status") or "unknown"

            failure_type_summary[failure_type] = failure_type_summary.get(failure_type, 0) + 1
            family_status_summary[family_status] = family_status_summary.get(family_status, 0) + 1
            outcome_status_summary[outcome_status] = outcome_status_summary.get(outcome_status, 0) + 1

            for action in memory.get("allowed_actions", []):
                bucket = action_effectiveness.setdefault(failure_type, {}).setdefault(
                    action,
                    {},
                )
                bucket[outcome_status] = bucket.get(outcome_status, 0) + 1

        return {
            "log_path": log_path,
            "entries": entries,
            "failure_type_summary": failure_type_summary,
            "family_status_summary": family_status_summary,
            "outcome_status_summary": outcome_status_summary,
            "action_effectiveness": action_effectiveness,
        }

    def infer_next_round_plan_from_legacy_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        best_candidate = payload.get("best_candidate") or {}
        batch_analysis = payload.get("batch_analysis") or {}
        batch_failure_counts = batch_analysis.get("batch_failure_counts") or {}
        plateau_info = batch_analysis.get("plateau_info")
        if not isinstance(plateau_info, dict):
            plateau_info = {
                "plateau_detected": bool(batch_analysis.get("plateau_detected")),
                "reason": batch_analysis.get("plateau_reason"),
            }

        synthetic_summary = {
            "alpha_id": best_candidate.get("alpha_id"),
            "sharpe": best_candidate.get("sharpe"),
            "fitness": best_candidate.get("fitness"),
            "turnover": best_candidate.get("turnover"),
            "checks": {
                "CONCENTRATED_WEIGHT": {
                    "result": "PASS" if best_candidate.get("stop_condition", {}).get("details", {}).get("all_required_pass") else None,
                    "value": None,
                },
                "IS_LADDER_SHARPE": {
                    "result": "PASS" if best_candidate.get("stop_condition", {}).get("details", {}).get("all_required_pass") else None,
                    "value": best_candidate.get("ladder_value"),
                },
                "LOW_SUB_UNIVERSE_SHARPE": {
                    "result": "PASS" if best_candidate.get("stop_condition", {}).get("details", {}).get("all_required_pass") else None,
                    "value": None,
                },
                "PROD_CORRELATION": {
                    "result": "PASS" if best_candidate.get("stop_condition", {}).get("details", {}).get("prod_ok") else "FAIL",
                    "value": best_candidate.get("prod_corr"),
                },
                "SELF_CORRELATION": {
                    "result": None,
                    "value": best_candidate.get("self_corr"),
                },
            },
        }
        return self.plan_next_round(
            best_summary=synthetic_summary,
            batch_failure_counts=batch_failure_counts,
            plateau_info=plateau_info,
        )

    def backfill_research_memory_log(
        self,
        log_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        source = Path(log_path)
        if not source.is_file():
            raise FileNotFoundError(log_path)

        target = Path(output_path) if output_path else source
        updated_lines: List[str] = []
        total = 0
        enriched = 0

        for raw_line in source.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            total += 1
            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError:
                updated_lines.append(raw_line)
                continue

            payload = line.get("payload", {}) or {}
            if "research_memory" in payload:
                updated_lines.append(json.dumps(line, ensure_ascii=False))
                continue

            if "best_candidate" in payload and "batch_analysis" in payload:
                automation_context = payload.get("automation_context") or self.build_automation_brief()
                next_round_plan = payload.get("next_round_plan") or self.infer_next_round_plan_from_legacy_payload(payload)
                payload["automation_context"] = automation_context
                payload["next_round_plan"] = next_round_plan
                payload["research_memory"] = self.build_research_memory_entry(payload)
                line["payload"] = payload
                enriched += 1
            elif payload.get("round") and (payload.get("theme") or payload.get("hypothesis") or payload.get("base_alpha")):
                theme_parts = [
                    str(payload.get("theme") or ""),
                    " ".join(payload.get("hypothesis", []) or []),
                    str(payload.get("notes") or ""),
                ]
                failure_type, primary_goal = _infer_goal_from_text(" ".join(part for part in theme_parts if part))
                action_library = self.action_library_for_failure_type(failure_type)
                settings_map = _as_mapping(payload.get("settings"))
                payload["automation_context"] = payload.get("automation_context") or self.build_automation_brief()
                payload["next_round_plan"] = payload.get("next_round_plan") or {
                    "primary_goal": primary_goal,
                    "failure_type": failure_type,
                    "reason": "Backfilled from legacy planning log theme and hypothesis.",
                    "allowed_actions": action_library["allowed"],
                    "blocked_actions": action_library["blocked"],
                    "single_goal_enforced": self.config.automation.single_round_goal_required,
                    "family_status": "active",
                    "frozen_directions": list(self.config.automation.frozen_directions),
                    "active_family": self.config.automation.active_family,
                    "batch_failure_counts": {},
                }
                payload["research_memory"] = {
                    "round": payload.get("round"),
                    "region": settings_map.get("region") or payload.get("region"),
                    "universe": settings_map.get("universe") or payload.get("universe"),
                    "delay": settings_map.get("delay") or payload.get("delay"),
                    "neutralization": settings_map.get("neutralization") or payload.get("neutralization"),
                    "active_family": ((payload.get("automation_context") or {}).get("active_family")),
                    "frozen_directions": ((payload.get("automation_context") or {}).get("frozen_directions")) or [],
                    "best_alpha_id": payload.get("best_alpha") or payload.get("base_alpha"),
                    "best_metrics": payload.get("metrics", {}),
                    "failure_type": failure_type,
                    "primary_goal": primary_goal,
                    "family_status": "active",
                    "allowed_actions": action_library["allowed"],
                    "blocked_actions": action_library["blocked"],
                    "outcome_status": "planning_only",
                    "outcome_summary": "Backfilled from a legacy planning log entry without batch-result payload.",
                    "batch_failure_counts": {},
                    "plateau_info": None,
                    "notes": payload.get("notes", []),
                }
                line["payload"] = payload
                enriched += 1

            updated_lines.append(json.dumps(line, ensure_ascii=False))

        target.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
        return {
            "source_log": str(source),
            "output_log": str(target),
            "total_lines": total,
            "enriched_lines": enriched,
        }

    def analyze_batch_outcomes(self, alpha_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        analyzed = [self.diagnose_alpha_summary(summary) for summary in alpha_summaries]
        failing_metrics: Dict[str, int] = {}
        for item in analyzed:
            for diagnosis in item["diagnoses"]:
                if diagnosis["status"] != "FAIL":
                    continue
                metric = diagnosis["metric"]
                failing_metrics[metric] = failing_metrics.get(metric, 0) + 1

        plateau_info = self.detect_plateau(alpha_summaries)

        return {
            "batch_failure_counts": failing_metrics,
            "plateau_detected": plateau_info.get("plateau_detected", False),
            "plateau_reason": plateau_info.get("reason"),
            "plateau_info": plateau_info,
            "next_step_bias": self.recommend_next_step_bias(
                failing_metrics,
                plateau_info.get("plateau_detected", False),
            ),
            "per_alpha_diagnosis": analyzed,
        }

    def recommend_next_step_bias(self, failing_metrics: Dict[str, int], plateau_detected: bool) -> Dict[str, Any]:
        stability_metrics = {"IS_LADDER_SHARPE", "LOW_2Y_SHARPE"}
        if not failing_metrics:
            return {
                "mode": "submit_or_target_band_refine",
                "reason": "Core workflow floors are passing.",
                "actions": [
                    "Run submission correlation checks if still pending.",
                    "Only refine toward target fitness/correlation if the current alpha is already acceptable.",
                ],
            }

        if plateau_detected and any(metric in failing_metrics for metric in stability_metrics):
            return {
                "mode": "branch_shape_substitution",
                "reason": "Ladder remains the blocker and local parameter scans are plateaued.",
                "actions": [
                    "Keep the successful branch fixed.",
                    "Replace only the likely offending branch shape.",
                    "Try same-field time-shape substitutions before changing core fields.",
                    "Review literature, forum examples, and stored experience before approving another nearby retry.",
                ],
            }

        if "PROD_CORRELATION" in failing_metrics and len(failing_metrics) == 1:
            return {
                "mode": "local_decorrelation",
                "reason": "Only production correlation remains problematic.",
                "actions": [
                    "Decorrelate the smallest possible branch.",
                    "Prefer auxiliary-field substitution or branch-local shape change.",
                    "Avoid full-expression rewrites.",
                ],
            }

        if len(failing_metrics) == 1 and next(iter(failing_metrics)) in stability_metrics:
            return {
                "mode": "stability_repair",
                "reason": "Only cross-period stability remains problematic.",
                "actions": [
                    "Treat the issue as a time-shape and stability problem.",
                    "Prefer scale or normalization alternatives if rank-like branches are stuck.",
                    "Do not keep grinding nearby parameter values once a plateau is visible.",
                    "If local options look exhausted, expand the hypothesis set through literature and peer cases first.",
                ],
            }

        if plateau_detected:
            return {
                "mode": "external_research_refresh",
                "reason": "The family looks locally plateaued; repeated nearby retries are likely to loop on the same bottleneck.",
                "actions": [
                    "Pause blind local tweaking.",
                    "Review papers, forum posts, and stored mining lessons for adjacent structures or datasets.",
                    "Return with a new hypothesis family rather than another cosmetic variant batch.",
                ],
            }

        return {
            "mode": "structure_rebuild",
            "reason": "Multiple core metrics are still failing.",
            "actions": [
                "Return to structure-level search.",
                "Do not enter fine-tuning.",
                "Re-validate field logic and branch responsibilities.",
                "If this line is looping, use external research to widen the family search before trying more local edits.",
            ],
        }

    def compare_against_existing(
        self,
        expressions: Sequence[str],
        existing_alpha_records: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        comparisons: List[Dict[str, Any]] = []
        for expression in expressions:
            current_fp = fingerprint_expression(expression)
            similar: List[Dict[str, Any]] = []
            for alpha in existing_alpha_records:
                code = ((alpha.get("regular", {}) or {}).get("code")) or ""
                if not code:
                    continue
                similarity = compare_fingerprints(current_fp, fingerprint_expression(code))
                similar.append(
                    {
                        "id": alpha.get("id"),
                        "similarity": similarity,
                    }
                )
            similar.sort(
                key=lambda item: (
                    -item["similarity"]["same_skeleton"],
                    -item["similarity"]["operator_jaccard"],
                    -item["similarity"]["field_jaccard"],
                )
            )
            comparisons.append(
                {
                    "expression": expression,
                    "top_matches": similar[:5],
                }
            )
        return {"comparisons": comparisons}

    def build_filter_thresholds(self) -> FilterThresholds:
        return FilterThresholds(
            min_sharpe=self.config.targets.min_sharpe,
            min_fitness=self.config.targets.min_fitness,
            min_turnover=self.config.targets.min_turnover,
            max_turnover=self.config.targets.max_turnover,
            max_prod_correlation=self.config.targets.max_prod_correlation,
            max_self_correlation=self.config.targets.max_self_correlation,
            require_prod_pass=False,
        )

    def infer_batch_settings(self, platform_alphas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not platform_alphas:
            return self.normalized_settings()

        settings = (platform_alphas[0].get("settings", {}) or {})
        return {
            "instrument_type": settings.get("instrumentType", self.config.instrument_type),
            "region": settings.get("region", self.normalized_settings()["region"]),
            "universe": settings.get("universe", self.config.universe),
            "delay": settings.get("delay", self.config.delay),
            "neutralization": settings.get("neutralization", self.config.neutralization),
            "language": settings.get("language", self.config.language),
        }

    def normalize_platform_alpha(self, alpha: Dict[str, Any]) -> Dict[str, Any]:
        is_data = alpha.get("is", {}) or {}
        checks = {}
        for check in is_data.get("checks", []) or []:
            name = check.get("name")
            if not name:
                continue
            checks[name] = {
                "result": check.get("result"),
                "value": check.get("value"),
                "year": check.get("year"),
                "startDate": check.get("startDate"),
                "endDate": check.get("endDate"),
            }

        prod_correlation = is_data.get("prodCorrelation")
        self_correlation = is_data.get("selfCorrelation")
        if isinstance(prod_correlation, (int, float)):
            prod_check = checks.get("PROD_CORRELATION", {})
            checks["PROD_CORRELATION"] = {
                "result": prod_check.get("result"),
                "value": prod_correlation,
                "year": prod_check.get("year"),
                "startDate": prod_check.get("startDate"),
                "endDate": prod_check.get("endDate"),
            }
        if isinstance(self_correlation, (int, float)):
            self_check = checks.get("SELF_CORRELATION", {})
            checks["SELF_CORRELATION"] = {
                "result": self_check.get("result"),
                "value": self_correlation,
                "year": self_check.get("year"),
                "startDate": self_check.get("startDate"),
                "endDate": self_check.get("endDate"),
            }

        return {
            "alpha_id": alpha.get("id"),
            "id": alpha.get("id"),
            "expression": ((alpha.get("regular", {}) or {}).get("code")),
            "code": ((alpha.get("regular", {}) or {}).get("code")),
            "operatorCount": ((alpha.get("regular", {}) or {}).get("operatorCount")),
            "sharpe": is_data.get("sharpe"),
            "fitness": is_data.get("fitness"),
            "turnover": is_data.get("turnover"),
            "returns": is_data.get("returns"),
            "drawdown": is_data.get("drawdown"),
            "margin": is_data.get("margin"),
            "checks": {
                "CONCENTRATED_WEIGHT": checks.get("CONCENTRATED_WEIGHT", {"result": None, "value": None}),
                "IS_LADDER_SHARPE": checks.get("IS_LADDER_SHARPE", {"result": None, "value": None}),
                "LOW_2Y_SHARPE": checks.get("LOW_2Y_SHARPE", {"result": None, "value": None}),
                "LOW_SUB_UNIVERSE_SHARPE": checks.get("LOW_SUB_UNIVERSE_SHARPE", {"result": None, "value": None}),
                "PROD_CORRELATION": checks.get("PROD_CORRELATION", {"result": None, "value": None}),
                "SELF_CORRELATION": checks.get("SELF_CORRELATION", {"result": None, "value": None}),
                "LOW_SHARPE": checks.get("LOW_SHARPE", {"result": None, "value": None}),
                "LOW_FITNESS": checks.get("LOW_FITNESS", {"result": None, "value": None}),
                "LOW_TURNOVER": checks.get("LOW_TURNOVER", {"result": None, "value": None}),
                "HIGH_TURNOVER": checks.get("HIGH_TURNOVER", {"result": None, "value": None}),
            },
            "settings": alpha.get("settings", {}) or {},
            "raw": alpha,
        }

    def evaluate_stop_condition(self, alpha_summary: Dict[str, Any]) -> Dict[str, Any]:
        checks = alpha_summary.get("checks", {})
        prod = checks.get("PROD_CORRELATION", {})
        self_corr_check = checks.get("SELF_CORRELATION", {})
        turnover = alpha_summary.get("turnover")
        fitness = alpha_summary.get("fitness")
        sharpe = alpha_summary.get("sharpe")
        margin = alpha_summary.get("margin")

        stability_name = _stability_check_name(alpha_summary)
        all_required_pass = (
            (not self.config.targets.require_weight_pass or checks.get("CONCENTRATED_WEIGHT", {}).get("result") == "PASS")
            and (not self.config.targets.require_stability_pass or _stability_is_passing(alpha_summary))
            and (not self.config.targets.require_sub_universe_pass or checks.get("LOW_SUB_UNIVERSE_SHARPE", {}).get("result") == "PASS")
        )
        prod_value = prod.get("value")
        self_corr_value = self_corr_check.get("value")
        prod_ok = isinstance(prod_value, (int, float)) and prod_value < self.config.targets.max_prod_correlation
        target_prod_hit = isinstance(prod_value, (int, float)) and prod_value < self.config.targets.target_prod_correlation
        self_corr_ok = isinstance(self_corr_value, (int, float)) and self_corr_value < self.config.targets.max_self_correlation
        target_self_corr_hit = isinstance(self_corr_value, (int, float)) and self_corr_value < self.config.targets.target_self_correlation
        fitness_ok = isinstance(fitness, (int, float)) and fitness >= self.config.targets.min_fitness
        target_fitness_hit = isinstance(fitness, (int, float)) and fitness >= self.config.targets.target_fitness
        sharpe_ok = isinstance(sharpe, (int, float)) and sharpe >= self.config.targets.min_sharpe
        target_sharpe_hit = isinstance(sharpe, (int, float)) and sharpe >= self.config.targets.target_sharpe
        turnover_ok = isinstance(turnover, (int, float)) and self.config.targets.min_turnover <= turnover <= self.config.targets.max_turnover
        margin_bps = float(margin) * 10000.0 if isinstance(margin, (int, float)) else None
        margin_ok = isinstance(margin_bps, (int, float)) and margin_bps >= self.config.targets.min_margin_bps

        done = all_required_pass and prod_ok and self_corr_ok and fitness_ok and sharpe_ok and turnover_ok
        return {
            "done": done,
            "meets_floor": done,
            "hits_target_band": done and target_sharpe_hit and target_fitness_hit and target_prod_hit and target_self_corr_hit and margin_ok,
            "details": {
                "all_required_pass": all_required_pass,
                "fitness_ok": fitness_ok,
                "target_fitness_hit": target_fitness_hit,
                "prod_ok": prod_ok,
                "target_prod_hit": target_prod_hit,
                "self_corr_ok": self_corr_ok,
                "target_self_corr_hit": target_self_corr_hit,
                "sharpe_ok": sharpe_ok,
                "target_sharpe_hit": target_sharpe_hit,
                "turnover_ok": turnover_ok,
                "margin_bps": margin_bps,
                "margin_ok": margin_ok,
                "stability_check_name": stability_name,
            },
        }

    def build_batch_log_payload(
        self,
        round_name: str,
        multisim_id: str,
        platform_alphas: Sequence[Dict[str, Any]],
        notes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        normalized = [self.normalize_platform_alpha(alpha) for alpha in platform_alphas]
        batch_settings = self.infer_batch_settings(platform_alphas)
        batch_analysis = self.analyze_batch_outcomes(normalized)
        stop_conditions = {
            summary.get("alpha_id"): self.evaluate_stop_condition(summary)
            for summary in normalized
            if summary.get("alpha_id")
        }
        floor_winners = [
            alpha_id
            for alpha_id, stop in stop_conditions.items()
            if stop.get("meets_floor")
        ]
        target_band_winners = [
            alpha_id
            for alpha_id, stop in stop_conditions.items()
            if stop.get("hits_target_band")
        ]
        qualified_goal_met = len(floor_winners) >= self.config.automation.min_qualified_alphas_per_day

        best_candidate: Optional[Dict[str, Any]] = None
        best_summary: Optional[Dict[str, Any]] = None
        best_score: Tuple[float, float, float, float] = (-1e9, -1e9, -1e9, -1e9)
        for summary in normalized:
            checks = summary.get("checks", {})
            stop = stop_conditions.get(summary.get("alpha_id"), {})
            floor_score = 1.0 if stop.get("meets_floor") else 0.0
            stability_check = _stability_check(summary)
            ladder_pass = 1.0 if _stability_is_passing(summary) else 0.0
            prod_val = checks.get("PROD_CORRELATION", {}).get("value")
            prod_score = -float(prod_val) if isinstance(prod_val, (int, float)) else 0.0
            fitness = float(summary.get("fitness")) if isinstance(summary.get("fitness"), (int, float)) else -1e9
            score = (floor_score, ladder_pass, fitness, prod_score)
            if score > best_score:
                best_score = score
                best_summary = summary
                best_candidate = {
                    "alpha_id": summary.get("alpha_id"),
                    "sharpe": summary.get("sharpe"),
                    "fitness": summary.get("fitness"),
                    "turnover": summary.get("turnover"),
                    "ladder_value": stability_check.get("value"),
                    "prod_corr": checks.get("PROD_CORRELATION", {}).get("value"),
                    "self_corr": checks.get("SELF_CORRELATION", {}).get("value"),
                    "operator_count": summary.get("operatorCount"),
                    "stop_condition": stop,
                }

        next_round_plan = self.plan_next_round(
            best_summary=best_summary,
            batch_failure_counts=batch_analysis["batch_failure_counts"],
            plateau_info=batch_analysis.get("plateau_info", {}),
        )

        results = []
        for summary in normalized:
            checks = summary.get("checks", {})
            stop = stop_conditions.get(summary.get("alpha_id"), {})
            classification = self.classify_failure_type(summary)
            results.append(
                {
                    "alpha_id": summary.get("alpha_id"),
                    "expression": summary.get("expression"),
                    "operator_count": summary.get("operatorCount"),
                    "sharpe": summary.get("sharpe"),
                    "fitness": summary.get("fitness"),
                    "turnover": summary.get("turnover"),
                    "sub_universe_pass": checks.get("LOW_SUB_UNIVERSE_SHARPE", {}).get("result") == "PASS",
                    "ladder_pass": _stability_is_passing(summary),
                    "ladder_value": _stability_check(summary).get("value"),
                    "concentrated_weight_pass": checks.get("CONCENTRATED_WEIGHT", {}).get("result") == "PASS",
                    "prod_corr": checks.get("PROD_CORRELATION", {}).get("value"),
                    "self_corr": checks.get("SELF_CORRELATION", {}).get("value"),
                    "meets_floor": stop.get("meets_floor", False),
                    "hits_target_band": stop.get("hits_target_band", False),
                    "failure_type": classification["failure_type"],
                    "primary_goal": classification["primary_goal"],
                }
            )

        if floor_winners and qualified_goal_met:
            batch_analysis["next_step_bias"] = {
                "mode": "submit_or_target_band_refine",
                "reason": "The batch met the daily consultant-grade output goal.",
                "actions": [
                    "Protect the winning structure.",
                    "Run submission checks before starting another rebuild cycle.",
                    "Only refine further if targeting a tighter fitness or correlation band.",
                    "Prefer follow-up mining in new pyramids rather than cloning the same family.",
                ],
            }
        elif floor_winners:
            batch_analysis["next_step_bias"] = {
                "mode": "qualified_output_gap",
                "reason": f"Only {len(floor_winners)} alpha(s) met consultant-grade floors; daily goal is {self.config.automation.min_qualified_alphas_per_day}.",
                "actions": [
                    "Keep the existing qualified winner as anchor.",
                    "Open the next batch in a diversified sibling or new pyramid family.",
                    "Bias toward low-correlation, deployable candidates rather than marginal fitness polish.",
                ],
            }
        elif best_summary is not None:
            best_diagnosis = self.diagnose_alpha_summary(best_summary)
            failing_metrics = [
                diagnosis["metric"]
                for diagnosis in best_diagnosis.get("diagnoses", [])
                if diagnosis.get("status") == "FAIL"
            ]
            if failing_metrics == ["PROD_CORRELATION"]:
                batch_analysis["next_step_bias"] = {
                    "mode": "local_decorrelation",
                    "reason": "The leading alpha only misses production correlation; the rest of the batch should not force a full rebuild.",
                    "actions": [
                        "Keep the winning structure intact.",
                        "Decorrelate the smallest responsible branch.",
                        "Re-run a focused branch-local batch instead of restarting structure search.",
                    ],
                }
            elif len(failing_metrics) == 1 and failing_metrics[0] in {"IS_LADDER_SHARPE", "LOW_2Y_SHARPE"}:
                batch_analysis["next_step_bias"] = {
                    "mode": "stability_repair",
                    "reason": "The leading alpha only misses cross-period stability; broad batch failures are secondary.",
                    "actions": [
                        "Preserve the current structure.",
                        "Change only the branch most responsible for time-shape instability.",
                        "Run a narrow ladder-focused variant batch.",
                    ],
                }

        return {
            "region": batch_settings["region"],
            "universe": batch_settings["universe"],
            "delay": batch_settings["delay"],
            "neutralization": batch_settings["neutralization"],
            "round": round_name,
            "multisim_id": multisim_id,
            "results": results,
            "best_candidate": best_candidate,
            "batch_analysis": batch_analysis,
            "next_round_plan": next_round_plan,
            "floor_winners": floor_winners,
            "target_band_winners": target_band_winners,
            "qualified_alpha_goal": self.config.automation.min_qualified_alphas_per_day,
            "qualified_alpha_goal_met": qualified_goal_met,
            "notes": list(notes or []),
            "automation_context": self.build_automation_brief(),
            "research_memory": self.build_research_memory_entry(
                {
                    "region": batch_settings["region"],
                    "universe": batch_settings["universe"],
                    "delay": batch_settings["delay"],
                    "neutralization": batch_settings["neutralization"],
                    "round": round_name,
                    "best_candidate": best_candidate,
                    "batch_analysis": batch_analysis,
                    "next_round_plan": next_round_plan,
                    "notes": list(notes or []),
                    "automation_context": self.build_automation_brief(),
                }
            ),
        }

    def extract_children_from_multisim(self, multisim_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        root = multisim_payload.get("result", multisim_payload)
        children = root.get("children", []) or []
        extracted: List[Dict[str, Any]] = []
        for child in children:
            extracted.append(
                {
                    "simulation_id": child.get("simulation_id") or child.get("id"),
                    "alpha_id": child.get("alpha_id") or (child.get("raw", {}) or {}).get("alpha"),
                    "expression": child.get("expression") or child.get("regular") or (child.get("raw", {}) or {}).get("regular"),
                    "status": child.get("status"),
                    "error": child.get("error"),
                    "settings": child.get("settings") or (child.get("raw", {}) or {}).get("settings") or {},
                    "raw": child,
                }
            )
        return extracted

    def build_batch_payload_from_multisim(
        self,
        round_name: str,
        multisim_payload: Dict[str, Any],
        platform_alphas: Sequence[Dict[str, Any]],
        notes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        root = multisim_payload.get("result", multisim_payload)
        multisim_id = root.get("multisim_id") or root.get("id") or ""
        children = self.extract_children_from_multisim(multisim_payload)
        alpha_map = {alpha.get("id"): alpha for alpha in platform_alphas if alpha.get("id")}

        matched_alphas: List[Dict[str, Any]] = []
        missing_alpha_ids: List[str] = []
        child_errors: List[Dict[str, Any]] = []

        for child in children:
            alpha_id = child.get("alpha_id")
            if child.get("status") != "COMPLETE" or child.get("error"):
                child_errors.append(
                    {
                        "simulation_id": child.get("simulation_id"),
                        "alpha_id": alpha_id,
                        "status": child.get("status"),
                        "error": child.get("error"),
                        "expression": child.get("expression"),
                    }
                )
            if not alpha_id:
                continue
            alpha = alpha_map.get(alpha_id)
            if alpha is None:
                missing_alpha_ids.append(alpha_id)
                continue
            matched_alphas.append(alpha)

        payload = self.build_batch_log_payload(
            round_name=round_name,
            multisim_id=multisim_id,
            platform_alphas=matched_alphas,
            notes=notes,
        )
        payload["multisim_children_count"] = len(children)
        payload["matched_alpha_count"] = len(matched_alphas)
        payload["missing_alpha_ids"] = missing_alpha_ids
        payload["child_errors"] = child_errors
        return payload

    def append_backtest_results(self, path: str, payload: Dict[str, Any]) -> None:
        timestamped = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(timestamped, ensure_ascii=False))
            handle.write("\n")

    def append_multisim_batch_results(
        self,
        path: str,
        round_name: str,
        multisim_payload: Dict[str, Any],
        platform_alphas: Sequence[Dict[str, Any]],
        notes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        payload = self.build_batch_payload_from_multisim(
            round_name=round_name,
            multisim_payload=multisim_payload,
            platform_alphas=platform_alphas,
            notes=notes,
        )
        self.append_backtest_results(path, payload)
        return payload

    def update_local_inventory(self, alpha: Dict[str, Any]) -> None:
        update_inventory_with_alpha(self.config.inventory_root, alpha)

    def update_local_inventory_many(self, alphas: Iterable[Dict[str, Any]]) -> None:
        for alpha in alphas:
            self.update_local_inventory(alpha)

    def summarize_platform_results(self, raw_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        return summarize_alphas(
            results=raw_results,
            target=TargetSettings(
                instrument_type=self.config.instrument_type,
                region=self.normalized_settings()["region"],
                universe=self.config.universe,
                delay=self.config.delay,
            ),
            thresholds=self.build_filter_thresholds(),
        )

    def build_workflow_snapshot(
        self,
        best_sharpe: Optional[float] = None,
        best_fitness: Optional[float] = None,
    ) -> Dict[str, Any]:
        stage = self.determine_stage(best_sharpe=best_sharpe, best_fitness=best_fitness)
        snapshot = {
            "settings": self.normalized_settings(),
            "research_brief": self.build_research_brief(),
            "automation_brief": self.build_automation_brief(),
            "performance_targets": asdict(self.config.targets),
            "integrated_heuristics": self.build_integrated_workflow_heuristics(),
            "stage": stage,
            "batch_blueprint": [asdict(item) for item in self.build_batch_blueprint(stage)],
        }
        memory_path = self.config.automation.research_memory_path
        if memory_path:
            snapshot["research_memory_summary"] = self.summarize_research_memory(memory_path)
        return snapshot


def save_workflow_snapshot(path: str, snapshot: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")


def load_workflow_config(path: str) -> WorkflowConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    research = ResearchPolicy(**payload.get("research", {}))
    research.require_economic_rationale_before_expression = True
    targets = PerformanceTargets(**payload.get("targets", {}))
    targets.min_sharpe = max(targets.min_sharpe, 1.8)
    targets.min_fitness = max(targets.min_fitness, 1.3)
    targets.max_prod_correlation = min(targets.max_prod_correlation, 0.55)
    targets.max_self_correlation = min(targets.max_self_correlation, 0.55)
    targets.target_prod_correlation = min(targets.target_prod_correlation, 0.40)
    targets.target_self_correlation = min(targets.target_self_correlation, 0.40)
    targets.target_sharpe = max(targets.target_sharpe, 2.2)
    targets.target_fitness = max(targets.target_fitness, 1.6)
    targets.min_margin_bps = max(targets.min_margin_bps, 4.0)
    targets.require_stability_pass = True
    targets.require_sub_universe_pass = True
    targets.require_weight_pass = True
    automation = AutomationPolicy(**payload.get("automation", {}))
    automation.min_qualified_alphas_per_day = max(automation.min_qualified_alphas_per_day, 2)
    automation.prioritize_pyramids = True
    automation.prioritize_income_multipliers = True
    automation.experience_library_path = (
        automation.experience_library_path
        or "docs/research/public_experience_library_template.json"
    )
    automation.stage_scope = automation.stage_scope or "current_quarter_only"
    if not automation.region_allocation:
        automation.region_allocation = {"EUR": 0.50, "MEA": 0.25, "ASI": 0.25}
    if not automation.pyramid_priority_targets:
        automation.pyramid_priority_targets = [
            "EUR:D1:Fundamental",
            "EUR:D1:News",
            "EUR:D1:Other",
            "EUR:D1:Sentiment",
            "MEA:D1:Fundamental",
            "MEA:D1:Analyst",
            "ASI:D1:Model",
            "ASI:D1:Analyst",
            "ASI:D1:Sentiment",
            "ASI:D1:Other",
        ]
    if not automation.pyramid_priority_table:
        automation.pyramid_priority_table = [
            {"target": "EUR:D1:Fundamental", "priority": 1, "multiplier": 1.2, "difficulty": "medium_low", "role": "primary"},
            {"target": "EUR:D1:News", "priority": 2, "multiplier": 1.5, "difficulty": "high", "role": "primary"},
            {"target": "EUR:D1:Other", "priority": 3, "multiplier": 1.6, "difficulty": "high", "role": "primary"},
            {"target": "EUR:D1:Sentiment", "priority": 4, "multiplier": 1.4, "difficulty": "medium_high", "role": "primary"},
            {"target": "MEA:D1:Fundamental", "priority": 5, "multiplier": 1.5, "difficulty": "medium_high", "role": "secondary"},
            {"target": "MEA:D1:Analyst", "priority": 6, "multiplier": 1.9, "difficulty": "very_high", "role": "secondary"},
            {"target": "ASI:D1:Model", "priority": 7, "multiplier": 1.3, "difficulty": "medium", "role": "secondary"},
            {"target": "ASI:D1:Analyst", "priority": 8, "multiplier": 1.4, "difficulty": "high", "role": "secondary"},
            {"target": "ASI:D1:Sentiment", "priority": 9, "multiplier": 1.5, "difficulty": "high", "role": "secondary"},
            {"target": "ASI:D1:Other", "priority": 10, "multiplier": 1.5, "difficulty": "very_high", "role": "secondary"},
        ]
    return WorkflowConfig(
        instrument_type=payload.get("instrument_type", "EQUITY"),
        region=payload.get("region", "USA"),
        universe=payload.get("universe", "TOP3000"),
        delay=payload.get("delay", 1),
        neutralization=payload.get("neutralization", "NONE"),
        language=payload.get("language", "FASTEXPR"),
        field_types=payload.get("field_types", []) or [],
        baseline_alpha_id=payload.get("baseline_alpha_id"),
        baseline_expression=payload.get("baseline_expression"),
        field_catalog_path=payload.get("field_catalog_path"),
        max_operator_count=payload.get("max_operator_count", 8),
        batch_size=payload.get("batch_size", 8),
        inventory_root=payload.get("inventory_root", "alpha_inventory"),
        credentials_path=payload.get("credentials_path", "credential.txt"),
        operator_cache_path=payload.get("operator_cache_path", "operator_compatibility_cache.json"),
        research=research,
        targets=targets,
        automation=automation,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Workflow scaffold for WQ BRAIN alpha mining.")
    parser.add_argument("--config", help="Path to workflow config JSON.")
    parser.add_argument("--snapshot-out", help="Path to write workflow snapshot JSON.")
    parser.add_argument("--expr", action="append", default=[], help="Candidate expression to preflight.")
    parser.add_argument("--best-sharpe", type=float, help="Current best sharpe for stage gating.")
    parser.add_argument("--best-fitness", type=float, help="Current best fitness for stage gating.")
    parser.add_argument("--append-log-path", help="Append a workflow payload to a local log file.")
    parser.add_argument("--platform-alpha-json", action="append", default=[], help="Path to a saved platform alpha detail JSON file.")
    parser.add_argument("--multisim-children-json", help="Path to a saved get_multisim_children JSON file.")
    parser.add_argument("--batch-round-name", help="Round label for batch payload generation.")
    parser.add_argument("--multisim-id", help="Multisim id for batch payload generation.")
    parser.add_argument("--summarize-memory-log", help="Summarize research memory entries from a JSONL log.")
    parser.add_argument("--backfill-memory-log", help="Backfill research_memory entries into a JSONL log.")
    parser.add_argument("--backfill-output-path", help="Optional output path for a backfilled JSONL log.")
    args = parser.parse_args()

    config = load_workflow_config(args.config) if args.config else WorkflowConfig()
    runner = WorkflowRunner(config=config)

    snapshot = runner.build_workflow_snapshot(
        best_sharpe=args.best_sharpe,
        best_fitness=args.best_fitness,
    )
    if args.snapshot_out:
        save_workflow_snapshot(args.snapshot_out, snapshot)
    else:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))

    if args.expr:
        print(json.dumps(runner.preflight_batch(args.expr), indent=2, ensure_ascii=False))
    if args.append_log_path:
        runner.append_backtest_results(
            args.append_log_path,
            {
                "settings": runner.normalized_settings(),
                "research_brief": runner.build_research_brief(),
                "field_catalog": runner.field_catalog_status(),
            },
        )
    if args.platform_alpha_json:
        platform_alphas = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in args.platform_alpha_json
        ]
        if args.multisim_children_json and args.batch_round_name:
            multisim_payload = json.loads(Path(args.multisim_children_json).read_text(encoding="utf-8"))
            payload = runner.build_batch_payload_from_multisim(
                round_name=args.batch_round_name,
                multisim_payload=multisim_payload,
                platform_alphas=platform_alphas,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif args.batch_round_name and args.multisim_id:
            payload = runner.build_batch_log_payload(
                round_name=args.batch_round_name,
                multisim_id=args.multisim_id,
                platform_alphas=platform_alphas,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            normalized = [runner.normalize_platform_alpha(alpha) for alpha in platform_alphas]
            analysis = runner.analyze_batch_outcomes(normalized)
            print(json.dumps({"normalized": normalized, "analysis": analysis}, indent=2, ensure_ascii=False))
    if args.summarize_memory_log:
        print(json.dumps(runner.summarize_research_memory(args.summarize_memory_log), indent=2, ensure_ascii=False))
    if args.backfill_memory_log:
        print(
            json.dumps(
                runner.backfill_research_memory_log(
                    log_path=args.backfill_memory_log,
                    output_path=args.backfill_output_path,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
