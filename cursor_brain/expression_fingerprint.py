from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from expression_validator import (
    ASTNode,
    AssignmentNode,
    BinaryOpNode,
    BoolNode,
    FunctionCallNode,
    IdentifierNode,
    NanNode,
    NumberNode,
    Parser,
    ProgramNode,
    StringNode,
    Tokenizer,
    UnaryOpNode,
)


THEME_MAP = {
    "trade_when": "A",
    "keep": "A",
    "if_else": "A",
    "nan_mask": "A",
    "days_from_last_change": "B",
    "filter": "B",
    "group_backfill": "B",
    "hump": "B",
    "hump_decay": "B",
    "jump_decay": "B",
    "kth_element": "B",
    "last_diff_value": "B",
    "ts_backfill": "B",
    "clamp": "C",
    "left_tail": "C",
    "nan_out": "C",
    "pasteurize": "C",
    "purify": "C",
    "replace": "C",
    "right_tail": "C",
    "tail": "C",
    "truncate": "C",
    "winsorize": "C",
    "group_multi_regression": "D",
    "group_vector_neut": "D",
    "group_vector_proj": "D",
    "multi_regression": "D",
    "regression_neut": "D",
    "regression_proj": "D",
    "ts_poly_regression": "D",
    "ts_regression": "D",
    "ts_theilsen": "D",
    "ts_vector_neut": "D",
    "ts_vector_proj": "D",
    "vector_neut": "D",
    "vector_proj": "D",
    "ts_co_kurtosis": "E",
    "ts_co_skewness": "E",
    "ts_corr": "E",
    "ts_covariance": "E",
    "ts_partial_corr": "E",
    "ts_triple_corr": "E",
    "inst_pnl": "F",
    "inst_tvr": "F",
    "one_side": "F",
    "rank_by_side": "F",
    "scale": "F",
    "scale_down": "F",
    "ts_delta_limit": "F",
    "ts_target_tvr_decay": "F",
    "ts_target_tvr_delta_limit": "F",
    "ts_target_tvr_hump": "F",
}

INFIX_OPERATOR_MAP = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
    "<": "less",
    "<=": "less_equal",
    ">": "greater",
    ">=": "greater_equal",
    "==": "equal",
    "!=": "not_equal",
}

COMMON_OPERATOR_SET = {
    "ts_sum",
    "ts_mean",
    "rank",
    "zscore",
    "winsorize",
    "ts_std_dev",
    "scale",
    "round",
    "trade_when",
}


@dataclass
class ExpressionFingerprint:
    expression: str
    fields: List[str]
    operators: List[str]
    infix_operators: List[str]
    operator_count_estimate: int
    themes: List[str]
    theme_operators: List[str]
    common_operator_hits: List[str]
    assignment_vars: List[str]
    skeleton: str
    max_depth: int


def parse_expression(expression: str) -> ProgramNode:
    tokenizer = Tokenizer(expression)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


def _walk(node: ASTNode) -> Iterable[ASTNode]:
    yield node
    if isinstance(node, ProgramNode):
        for stmt in node.statements:
            yield from _walk(stmt)
        yield from _walk(node.final_expr)
    elif isinstance(node, AssignmentNode):
        yield from _walk(node.value)
    elif isinstance(node, BinaryOpNode):
        yield from _walk(node.left)
        yield from _walk(node.right)
    elif isinstance(node, UnaryOpNode):
        yield from _walk(node.operand)
    elif isinstance(node, FunctionCallNode):
        for arg in node.args:
            yield from _walk(arg)
        for value in node.kwargs.values():
            yield from _walk(value)


def _compute_depth(node: ASTNode) -> int:
    if isinstance(node, ProgramNode):
        depths = [_compute_depth(stmt) for stmt in node.statements]
        depths.append(_compute_depth(node.final_expr))
        return max(depths) if depths else 1
    if isinstance(node, AssignmentNode):
        return 1 + _compute_depth(node.value)
    if isinstance(node, BinaryOpNode):
        return 1 + max(_compute_depth(node.left), _compute_depth(node.right))
    if isinstance(node, UnaryOpNode):
        return 1 + _compute_depth(node.operand)
    if isinstance(node, FunctionCallNode):
        child_depths = [_compute_depth(arg) for arg in node.args]
        child_depths.extend(_compute_depth(value) for value in node.kwargs.values())
        return 1 + (max(child_depths) if child_depths else 0)
    return 1


def _node_to_skeleton(node: ASTNode) -> str:
    if isinstance(node, ProgramNode):
        stmt_parts = [_node_to_skeleton(stmt) for stmt in node.statements]
        stmt_parts.append(_node_to_skeleton(node.final_expr))
        return ";".join(stmt_parts)
    if isinstance(node, AssignmentNode):
        return f"assign({_node_to_skeleton(node.value)})"
    if isinstance(node, FunctionCallNode):
        arg_parts = [_node_to_skeleton(arg) for arg in node.args]
        kw_parts = [f"{key}={_node_to_skeleton(value)}" for key, value in sorted(node.kwargs.items())]
        return f"{node.name.lower()}({','.join(arg_parts + kw_parts)})"
    if isinstance(node, BinaryOpNode):
        op_name = INFIX_OPERATOR_MAP.get(node.op, node.op)
        return f"{op_name}({_node_to_skeleton(node.left)},{_node_to_skeleton(node.right)})"
    if isinstance(node, UnaryOpNode):
        return f"unary_{node.op}({_node_to_skeleton(node.operand)})"
    if isinstance(node, IdentifierNode):
        return "field"
    if isinstance(node, NumberNode):
        return "number"
    if isinstance(node, StringNode):
        return "string"
    if isinstance(node, BoolNode):
        return "bool"
    if isinstance(node, NanNode):
        return "nan"
    return node.__class__.__name__.lower()


def fingerprint_expression(expression: str) -> ExpressionFingerprint:
    ast = parse_expression(expression)
    assigned_vars = {stmt.var_name for stmt in ast.statements}
    fields: Set[str] = set()
    operators: Set[str] = set()
    infix_ops: Set[str] = set()

    for node in _walk(ast):
        if isinstance(node, FunctionCallNode):
            operators.add(node.name.lower())
        elif isinstance(node, BinaryOpNode):
            mapped = INFIX_OPERATOR_MAP.get(node.op)
            if mapped:
                operators.add(mapped)
                infix_ops.add(mapped)
        elif isinstance(node, UnaryOpNode) and node.op == "-":
            operators.add("reverse")
        elif isinstance(node, IdentifierNode) and node.name not in assigned_vars:
            fields.add(node.name)

    theme_ops = sorted(op for op in operators if op in THEME_MAP)
    themes = sorted({THEME_MAP[op] for op in theme_ops})
    common_hits = sorted(op for op in operators if op in COMMON_OPERATOR_SET)

    return ExpressionFingerprint(
        expression=expression,
        fields=sorted(fields),
        operators=sorted(operators),
        infix_operators=sorted(infix_ops),
        operator_count_estimate=len(operators),
        themes=themes,
        theme_operators=theme_ops,
        common_operator_hits=common_hits,
        assignment_vars=sorted(assigned_vars),
        skeleton=_node_to_skeleton(ast),
        max_depth=_compute_depth(ast),
    )


def compare_fingerprints(
    left: ExpressionFingerprint,
    right: ExpressionFingerprint,
) -> Dict[str, Any]:
    left_fields = set(left.fields)
    right_fields = set(right.fields)
    left_ops = set(left.operators)
    right_ops = set(right.operators)
    left_themes = set(left.themes)
    right_themes = set(right.themes)

    return {
        "field_jaccard": jaccard_similarity(left_fields, right_fields),
        "operator_jaccard": jaccard_similarity(left_ops, right_ops),
        "theme_jaccard": jaccard_similarity(left_themes, right_themes),
        "same_skeleton": left.skeleton == right.skeleton,
        "shared_fields": sorted(left_fields & right_fields),
        "shared_operators": sorted(left_ops & right_ops),
        "shared_themes": sorted(left_themes & right_themes),
    }


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def fingerprint_alpha_record(alpha: Dict[str, Any]) -> Dict[str, Any]:
    code = ((alpha.get("regular", {}) or {}).get("code")) or ""
    fingerprint = fingerprint_expression(code)
    payload = asdict(fingerprint)
    payload["id"] = alpha.get("id")
    payload["region"] = ((alpha.get("settings", {}) or {}).get("region"))
    payload["universe"] = ((alpha.get("settings", {}) or {}).get("universe"))
    payload["delay"] = ((alpha.get("settings", {}) or {}).get("delay"))
    return payload


def rank_novelty(
    expression: str,
    existing: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    target = fingerprint_expression(expression)
    ranked: List[Dict[str, Any]] = []
    for alpha in existing:
        code = ((alpha.get("regular", {}) or {}).get("code")) or ""
        if not code:
            continue
        baseline = fingerprint_expression(code)
        similarity = compare_fingerprints(target, baseline)
        ranked.append(
            {
                "id": alpha.get("id"),
                "similarity": similarity,
                "operator_count_estimate": baseline.operator_count_estimate,
                "themes": baseline.themes,
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["similarity"]["same_skeleton"],
            -item["similarity"]["operator_jaccard"],
            -item["similarity"]["field_jaccard"],
            -item["similarity"]["theme_jaccard"],
        )
    )
    return ranked


def save_fingerprint_report(path: str, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

