"""
Alpha表达式验证器
用于验证 WorldQuant BRAIN alpha 表达式的语法、字段/算子合法性、参数签名、以及算子作用域 Scope。

Scope 规则（按你的要求）：
1) 最终表达式必须落在 REGULAR 作用域（即最终“外层算子”必须支持 REGULAR；纯字段/常数视为允许）。
2) 任何用到的算子都必须支持 COMBO/REGULAR（不允许只属于 SELECTION 的算子）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import pandas as pd


# ============================================================================
# 第一部分：词法分析器（Tokenizer）
# ============================================================================


class TokenType(Enum):
    # 字面量
    NUMBER = "NUMBER"  # 123, 0.5, 1.23
    STRING = "STRING"  # "gaussian"
    BOOL = "BOOL"  # true, false
    NAN = "NAN"  # nan

    # 标识符
    IDENTIFIER = "IDENTIFIER"  # close, open, rank, my_var

    # 运算符
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    LESS = "LESS"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER = "GREATER"
    GREATER_EQUAL = "GREATER_EQUAL"
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"

    # 分隔符
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    ASSIGN = "ASSIGN"

    # 结束
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: Any
    position: int = 0


class Tokenizer:
    def __init__(self, expression: str):
        self.expression = expression
        self.pos = 0
        self.current_char = expression[0] if expression else None

    def error(self, msg: str):
        raise SyntaxError(f"Lexical error at position {self.pos}: {msg}")

    def advance(self):
        self.pos += 1
        if self.pos < len(self.expression):
            self.current_char = self.expression[self.pos]
        else:
            self.current_char = None

    def peek(self, offset: int = 1) -> Optional[str]:
        peek_pos = self.pos + offset
        if peek_pos < len(self.expression):
            return self.expression[peek_pos]
        return None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def read_number(self) -> Token:
        start_pos = self.pos
        num_str = ""
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == "."):
            num_str += self.current_char
            self.advance()

        try:
            value = float(num_str) if "." in num_str else int(num_str)
            return Token(TokenType.NUMBER, value, start_pos)
        except ValueError:
            self.error(f"Invalid number: {num_str}")

    def read_identifier(self) -> Token:
        start_pos = self.pos
        ident = ""
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            ident += self.current_char
            self.advance()

        ident_lower = ident.lower()
        if ident_lower == "true":
            return Token(TokenType.BOOL, True, start_pos)
        if ident_lower == "false":
            return Token(TokenType.BOOL, False, start_pos)
        if ident_lower == "nan":
            return Token(TokenType.NAN, float("nan"), start_pos)
        return Token(TokenType.IDENTIFIER, ident, start_pos)

    def read_string(self, quote_char: str) -> Token:
        start_pos = self.pos
        self.advance()  # skip opening quote
        string_val = ""
        while self.current_char is not None and self.current_char != quote_char:
            string_val += self.current_char
            self.advance()

        if self.current_char != quote_char:
            self.error(f"Unterminated string (expected closing {quote_char})")

        self.advance()  # skip closing quote
        return Token(TokenType.STRING, string_val, start_pos)

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # 负数：如果 '-' 后跟数字，且 '-' 前面是“开始/左括号/逗号/赋值”，就把它当作负号并合并成数字
            if self.current_char == "-":
                next_char = self.peek()
                if next_char and (next_char.isdigit() or next_char == "."):
                    prev_ok = not tokens or tokens[-1].type in {
                        TokenType.LPAREN,
                        TokenType.COMMA,
                        TokenType.ASSIGN,
                    }
                    if prev_ok:
                        self.advance()
                        num_token = self.read_number()
                        num_token.value = -num_token.value
                        tokens.append(num_token)
                        continue

            if self.current_char.isdigit() or (self.current_char == "." and self.peek() and self.peek().isdigit()):
                tokens.append(self.read_number())
                continue

            if self.current_char.isalpha() or self.current_char == "_":
                tokens.append(self.read_identifier())
                continue

            if self.current_char == '"':
                tokens.append(self.read_string('"'))
                continue
            if self.current_char == "'":
                tokens.append(self.read_string("'"))
                continue

            # 双字符运算符
            if self.current_char == "<" and self.peek() == "=":
                tokens.append(Token(TokenType.LESS_EQUAL, "<=", self.pos))
                self.advance()
                self.advance()
                continue
            if self.current_char == ">" and self.peek() == "=":
                tokens.append(Token(TokenType.GREATER_EQUAL, ">=", self.pos))
                self.advance()
                self.advance()
                continue
            if self.current_char == "=" and self.peek() == "=":
                tokens.append(Token(TokenType.EQUAL, "==", self.pos))
                self.advance()
                self.advance()
                continue
            if self.current_char == "!" and self.peek() == "=":
                tokens.append(Token(TokenType.NOT_EQUAL, "!=", self.pos))
                self.advance()
                self.advance()
                continue

            char_map = {
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.MULTIPLY,
                "/": TokenType.DIVIDE,
                "<": TokenType.LESS,
                ">": TokenType.GREATER,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                ",": TokenType.COMMA,
                ";": TokenType.SEMICOLON,
                "=": TokenType.ASSIGN,
            }

            if self.current_char in char_map:
                ttype = char_map[self.current_char]
                tokens.append(Token(ttype, self.current_char, self.pos))
                self.advance()
                continue

            self.error(f"Unknown character: '{self.current_char}'")

        tokens.append(Token(TokenType.EOF, None, self.pos))
        return tokens


# ============================================================================
# 第二部分：AST 节点
# ============================================================================


class ASTNode:
    pass


class NumberNode(ASTNode):
    def __init__(self, value: float):
        self.value = value


class StringNode(ASTNode):
    def __init__(self, value: str):
        self.value = value


class BoolNode(ASTNode):
    def __init__(self, value: bool):
        self.value = value


class NanNode(ASTNode):
    pass


class IdentifierNode(ASTNode):
    def __init__(self, name: str):
        self.name = name


class BinaryOpNode(ASTNode):
    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right


class UnaryOpNode(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand


class FunctionCallNode(ASTNode):
    def __init__(self, name: str, args: List[ASTNode], kwargs: Dict[str, ASTNode]):
        self.name = name
        self.args = args
        self.kwargs = kwargs


class AssignmentNode(ASTNode):
    def __init__(self, var_name: str, value: ASTNode):
        self.var_name = var_name
        self.value = value


class ProgramNode(ASTNode):
    def __init__(self, statements: List[AssignmentNode], final_expr: ASTNode):
        self.statements = statements
        self.final_expr = final_expr


# ============================================================================
# 第三部分：Parser
# ============================================================================


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else Token(TokenType.EOF, None)

    def error(self, msg: str):
        raise SyntaxError(f"Syntax error at position {self.current_token.position}: {msg}")

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token(TokenType.EOF, None)

    def peek(self, offset: int = 1) -> Token:
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return Token(TokenType.EOF, None)

    def expect(self, token_type: TokenType):
        if self.current_token.type != token_type:
            self.error(f"Expected {token_type}, got {self.current_token.type}")
        self.advance()

    def is_assignment(self) -> bool:
        if self.current_token.type != TokenType.IDENTIFIER:
            return False
        next_token = self.peek()
        return next_token.type == TokenType.ASSIGN

    def parse(self) -> ProgramNode:
        statements: List[AssignmentNode] = []

        while self.current_token.type != TokenType.EOF and self.is_assignment():
            stmt = self.parse_assignment()
            statements.append(stmt)

            if self.current_token.type == TokenType.SEMICOLON:
                self.advance()
            else:
                self.error("Expected ';' after assignment")

        if self.current_token.type == TokenType.EOF:
            self.error("Expected expression after assignments")

        final_expr = self.parse_expression()

        if self.current_token.type == TokenType.SEMICOLON:
            self.error("Final expression cannot end with semicolon")

        if self.current_token.type != TokenType.EOF:
            self.error(f"Unexpected token: {self.current_token.type}")

        return ProgramNode(statements, final_expr)

    def parse_assignment(self) -> AssignmentNode:
        var_name = self.current_token.value
        self.advance()
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        return AssignmentNode(var_name, value)

    def parse_expression(self) -> ASTNode:
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        left = self.parse_additive()
        while self.current_token.type in {
            TokenType.LESS,
            TokenType.LESS_EQUAL,
            TokenType.GREATER,
            TokenType.GREATER_EQUAL,
            TokenType.EQUAL,
            TokenType.NOT_EQUAL,
        }:
            op_token = self.current_token
            self.advance()
            right = self.parse_additive()
            left = BinaryOpNode(left, op_token.value, right)
        return left

    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        while self.current_token.type in {TokenType.PLUS, TokenType.MINUS}:
            op_token = self.current_token
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op_token.value, right)
        return left

    def parse_multiplicative(self) -> ASTNode:
        left = self.parse_unary()
        while self.current_token.type in {TokenType.MULTIPLY, TokenType.DIVIDE}:
            op_token = self.current_token
            self.advance()
            right = self.parse_unary()
            left = BinaryOpNode(left, op_token.value, right)
        return left

    def parse_unary(self) -> ASTNode:
        if self.current_token.type == TokenType.MINUS:
            op_token = self.current_token
            self.advance()
            operand = self.parse_unary()
            return UnaryOpNode(op_token.value, operand)
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        token = self.current_token

        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(token.value)

        if token.type == TokenType.STRING:
            self.advance()
            return StringNode(token.value)

        if token.type == TokenType.BOOL:
            self.advance()
            return BoolNode(token.value)

        if token.type == TokenType.NAN:
            self.advance()
            return NanNode()

        if token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()
            if self.current_token.type == TokenType.LPAREN:
                return self.parse_function_call(name)
            return IdentifierNode(name)

        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        self.error(f"Unexpected token: {token.type}")

    def parse_function_call(self, func_name: str) -> FunctionCallNode:
        self.expect(TokenType.LPAREN)
        args: List[ASTNode] = []
        kwargs: Dict[str, ASTNode] = {}

        while self.current_token.type != TokenType.RPAREN:
            if self.current_token.type == TokenType.IDENTIFIER and self.peek().type == TokenType.ASSIGN:
                key = self.current_token.value
                self.advance()
                self.expect(TokenType.ASSIGN)
                value = self.parse_expression()
                kwargs[key] = value
            else:
                args.append(self.parse_expression())

            if self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token.type != TokenType.RPAREN:
                self.error("Expected ',' or ')' in function call")

        self.expect(TokenType.RPAREN)
        return FunctionCallNode(func_name, args, kwargs)


# ============================================================================
# 第四部分：Operator Spec（从 operators.json 加载）
# ============================================================================


class ParamType(Enum):
    MATRIX = "MATRIX"
    VECTOR = "VECTOR"
    GROUP = "GROUP"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    STRING = "STRING"
    ANY = "ANY"


@dataclass
class ParamSpec:
    name: str
    param_type: ParamType
    optional: bool = False
    default_value: Any = None
    value_constraint: Optional[Callable[[Any], bool]] = None

    def validate_value(self, value: Any) -> Tuple[bool, str]:
        if self.value_constraint and not self.value_constraint(value):
            return False, f"Value {value} does not meet constraint for parameter '{self.name}'"
        return True, ""


@dataclass
class OperatorSpec:
    name: str
    scopes: List[str]
    positional_params: List[ParamSpec]
    keyword_params: Dict[str, ParamSpec]
    variadic: bool = False
    return_type: ParamType = ParamType.MATRIX
    min_args: int = 0


def _parse_param_type(type_str: str) -> ParamType:
    return ParamType[type_str.upper()]


def _build_value_constraint(param_json: Dict[str, Any]) -> Optional[Callable[[Any], bool]]:
    # 允许值集合
    allowed = param_json.get("allowed")
    if isinstance(allowed, list):
        allowed_set = set(allowed)

        def _in_allowed(v: Any) -> bool:
            return v in allowed_set

        return _in_allowed

    # 简单数值约束：gt/gte/lt/lte
    constraint = param_json.get("constraint")
    if isinstance(constraint, dict):
        gt = constraint.get("gt")
        gte = constraint.get("gte")
        lt = constraint.get("lt")
        lte = constraint.get("lte")

        def _num(v: Any) -> bool:
            try:
                x = float(v)
            except Exception:
                return False
            if gt is not None and not (x > float(gt)):
                return False
            if gte is not None and not (x >= float(gte)):
                return False
            if lt is not None and not (x < float(lt)):
                return False
            if lte is not None and not (x <= float(lte)):
                return False
            return True

        return _num

    return None


class OperatorsRepository:
    def __init__(self, operators_json_path: str = "operators.json"):
        operators_json_file = Path(operators_json_path)
        if not operators_json_file.is_file():
            # 允许相对当前脚本目录
            operators_json_file = Path(__file__).parent / operators_json_path
        if not operators_json_file.is_file():
            raise FileNotFoundError(f"operators.json not found at '{operators_json_path}'")

        raw = json.loads(operators_json_file.read_text(encoding="utf-8"))
        self.operators: Dict[str, OperatorSpec] = {}

        for op in raw.get("operators", []) or []:
            name = str(op["name"])
            scopes = list(op.get("scopes", []) or [])
            return_type = _parse_param_type(op.get("return_type", "MATRIX"))
            variadic = bool(op.get("variadic", False))
            min_args = int(op.get("min_args", 0) or 0)

            positional_params: List[ParamSpec] = []
            for p in op.get("positional_params", []) or []:
                p_name = p.get("name")
                p_type = _parse_param_type(p["type"])
                optional = bool(p.get("optional", False))
                default_value = p.get("default", None)
                value_constraint = _build_value_constraint(p)
                positional_params.append(
                    ParamSpec(
                        name=str(p_name),
                        param_type=p_type,
                        optional=optional,
                        default_value=default_value,
                        value_constraint=value_constraint,
                    )
                )

            keyword_params: Dict[str, ParamSpec] = {}
            for key, pv in (op.get("keyword_params", {}) or {}).items():
                p_type = _parse_param_type(pv["type"])
                optional = bool(pv.get("optional", False))
                default_value = pv.get("default", None)
                value_constraint = _build_value_constraint(pv)
                keyword_params[str(key)] = ParamSpec(
                    name=str(key),
                    param_type=p_type,
                    optional=optional,
                    default_value=default_value,
                    value_constraint=value_constraint,
                )

            self.operators[name.lower()] = OperatorSpec(
                name=name,
                scopes=scopes,
                positional_params=positional_params,
                keyword_params=keyword_params,
                variadic=variadic,
                return_type=return_type,
                min_args=min_args,
            )

        self._register_supplemental_vector_operators()

    def _register_supplemental_vector_operators(self) -> None:
        # Local operators.json lags behind the platform for some vec_* reducers.
        # Add the reducers we have already verified on-platform so preflight
        # stops emitting false "Unknown operator" errors.
        vector_to_matrix = [
            "vec_count",
            "vec_stddev",
            "vec_sum",
            "vec_avg",
            "vec_max",
            "vec_min",
            "vec_range",
            "vec_skewness",
            "vec_kurtosis",
            "vec_norm",
            "vec_ir",
        ]
        for name in vector_to_matrix:
            if name in self.operators:
                continue
            self.operators[name] = OperatorSpec(
                name=name,
                scopes=["COMBO", "REGULAR"],
                positional_params=[ParamSpec(name="x", param_type=ParamType.VECTOR)],
                keyword_params={},
                variadic=False,
                return_type=ParamType.MATRIX,
                min_args=1,
            )

    def get_operator_spec(self, name: str) -> Optional[OperatorSpec]:
        return self.operators.get(name.lower())

    def is_operator(self, name: str) -> bool:
        return name.lower() in self.operators


# ============================================================================
# 第五部分：数据上下文（Data Context）
# ============================================================================


def resolve_field_catalog_path(csv_path: Optional[str] = None) -> Optional[Path]:
    if csv_path:
        candidate = Path(csv_path)
        if candidate.is_file():
            return candidate
        local_candidate = Path(__file__).parent / csv_path
        if local_candidate.is_file():
            return local_candidate
        return None

    field_catalog_dir = Path(__file__).parent / "field_catalogs"
    if field_catalog_dir.is_dir():
        csv_files = sorted(field_catalog_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]

    fallback = Path(__file__).parent / "USA_TOP1000_1.csv"
    if fallback.is_file():
        return fallback
    return None


class DataContext:
    def __init__(
        self,
        csv_path: Optional[str] = None,
        operators_json_path: str = "operators.json",
    ):
        self.field_catalog_path = resolve_field_catalog_path(csv_path)
        self.datafields = self._load_datafields(self.field_catalog_path)
        self.field_catalog_loaded = bool(self.datafields)
        self.field_catalog_degraded = not self.field_catalog_loaded
        self.operators_repo = OperatorsRepository(operators_json_path=operators_json_path)

    def _load_datafields(self, csv_file: Optional[Path]) -> Dict[str, Dict[str, Any]]:
        csv_file = csv_file
        if csv_file is None or not csv_file.is_file():
            # 允许相对脚本目录
            return {}
        if not csv_file.is_file():
            # CSV 可能尚未由用户生成；为了不阻断表达式 Scope 校验流程，
            # 这里允许缺失 CSV，字段名将被当作 unknown（见 allow_unknown_fields）。
            return {}

        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        datafields: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            field_id = str(row["id"])
            datafields[field_id] = {
                "type": str(row["type"]).upper(),  # MATRIX/VECTOR/GROUP
                "description": row.get("description", "") if "description" in row else "",
            }
        return datafields

    def is_datafield(self, name: str) -> bool:
        name_lower = name.lower()
        return any(field.lower() == name_lower for field in self.datafields)

    def get_datafield_type(self, name: str) -> Optional[ParamType]:
        name_lower = name.lower()
        for field_name, field_info in self.datafields.items():
            if field_name.lower() == name_lower:
                return ParamType[field_info["type"]]
        return None

    def is_operator(self, name: str) -> bool:
        return self.operators_repo.is_operator(name)

    def get_operator_spec(self, name: str) -> Optional[OperatorSpec]:
        return self.operators_repo.get_operator_spec(name)

    def field_catalog_status(self) -> Dict[str, Any]:
        return {
            "field_catalog_path": str(self.field_catalog_path) if self.field_catalog_path else None,
            "field_catalog_loaded": self.field_catalog_loaded,
            "field_catalog_degraded": self.field_catalog_degraded,
            "field_count": len(self.datafields),
        }


# ============================================================================
# 第六部分：语义分析器（Semantic Analyzer）
# ============================================================================


class SemanticAnalyzer:
    def __init__(
        self,
        context: DataContext,
        allowed_scopes: Set[str],
        required_root_scope: str = "REGULAR",
        allow_unknown_fields: bool = True,
    ):
        self.context = context
        self.allowed_scopes = set(allowed_scopes)
        self.required_root_scope = required_root_scope
        self.allow_unknown_fields = allow_unknown_fields

        self.defined_vars: Dict[str, ParamType] = {}
        self.used_vars: Set[str] = set()
        self.errors: List[str] = []

    def analyze(self, ast: ProgramNode) -> Tuple[bool, List[str]]:
        self.errors = []
        self.defined_vars = {}
        self.used_vars = set()

        for stmt in ast.statements:
            self._analyze_assignment(stmt)

        final_expr_type = self._analyze_expression(ast.final_expr)

        # 禁止最终返回 GROUP（regular alpha 不应返回分组对象）
        if final_expr_type == ParamType.GROUP:
            self.errors.append(
                "Final expression cannot return GROUP type. Use group_* operators to consume GROUP values."
            )

        # Scope 强约束：最终外层必须支持 required_root_scope
        root_scopes = self._infer_root_scopes(ast.final_expr)
        if root_scopes is not None and self.required_root_scope not in root_scopes:
            self.errors.append(
                f"Final expression must be in '{self.required_root_scope}' scope, but root scopes={sorted(root_scopes)}"
            )

        self._check_unused_variables()
        return (len(self.errors) == 0, self.errors)

    def _infer_root_scopes(self, node: ASTNode) -> Optional[Set[str]]:
        """
        返回最终表达式“外层”的算子允许作用域集合。
        - FunctionCallNode => 该算子的 scopes
        - BinaryOpNode/UnaryOpNode => 由中缀/一元运算映射到对应 operator scopes
        - 常量/变量/字段 => 允许返回 None（不强制）
        """
        if isinstance(node, FunctionCallNode):
            spec = self.context.get_operator_spec(node.name)
            if not spec:
                return None
            return set(spec.scopes)

        if isinstance(node, BinaryOpNode):
            op_name = self._map_infix_op_to_operator(node.op)
            if not op_name:
                return None
            spec = self.context.get_operator_spec(op_name)
            return set(spec.scopes) if spec else None

        if isinstance(node, UnaryOpNode):
            op_name = self._map_unary_op_to_operator(node.op)
            if not op_name:
                return None
            spec = self.context.get_operator_spec(op_name)
            return set(spec.scopes) if spec else None

        return None

    def _check_operator_scope_allowed(self, operator_name: str):
        spec = self.context.get_operator_spec(operator_name)
        if not spec:
            self.errors.append(f"Unknown operator: '{operator_name}'")
            return
        if not (set(spec.scopes) & self.allowed_scopes):
            self.errors.append(
                f"Operator '{operator_name}' scopes={spec.scopes} is not allowed by allowed_scopes={sorted(self.allowed_scopes)}"
            )

    def _analyze_assignment(self, node: AssignmentNode):
        var_name = node.var_name

        if self.context.is_operator(var_name):
            self.errors.append(f"Variable name '{var_name}' conflicts with operator")
            return
        if self.context.is_datafield(var_name):
            self.errors.append(f"Variable name '{var_name}' conflicts with datafield")
            return

        forbidden_names = {"delta", "sum", "covariance", "delay"}
        if var_name.lower() in forbidden_names:
            self.errors.append(f"Variable name '{var_name}' is reserved and cannot be used")
            return

        expr_type = self._analyze_expression(node.value)
        self.defined_vars[var_name] = expr_type

    def _analyze_expression(self, node: ASTNode) -> ParamType:
        if isinstance(node, NumberNode):
            return ParamType.INT if isinstance(node.value, int) else ParamType.FLOAT

        if isinstance(node, StringNode):
            return ParamType.STRING

        if isinstance(node, BoolNode):
            return ParamType.BOOL

        if isinstance(node, NanNode):
            return ParamType.FLOAT

        if isinstance(node, IdentifierNode):
            return self._analyze_identifier(node)

        if isinstance(node, FunctionCallNode):
            return self._analyze_function_call(node)

        if isinstance(node, BinaryOpNode):
            # 先递归算子内部类型
            self._analyze_expression(node.left)
            self._analyze_expression(node.right)
            # 再做运算符作用域校验
            op_name = self._map_infix_op_to_operator(node.op)
            if op_name:
                self._check_operator_scope_allowed(op_name)
            return ParamType.MATRIX

        if isinstance(node, UnaryOpNode):
            self._analyze_expression(node.operand)
            op_name = self._map_unary_op_to_operator(node.op)
            if op_name:
                self._check_operator_scope_allowed(op_name)
            # unary '-' 作用域上等价于 reverse(x)，返回矩阵；其它一元运算如需要再扩展
            return ParamType.MATRIX

        self.errors.append(f"Unknown node type: {type(node).__name__}")
        return ParamType.ANY

    def _analyze_identifier(self, node: IdentifierNode) -> ParamType:
        name = node.name
        if name in self.defined_vars:
            self.used_vars.add(name)
            return self.defined_vars[name]
        if self.context.is_datafield(name):
            return self.context.get_datafield_type(name) or ParamType.ANY
        if self.allow_unknown_fields:
            # 不知道字段是否存在时，只做弱类型检查：允许继续解析并返回 ANY
            return ParamType.ANY
        self.errors.append(f"Undefined identifier '{name}'")
        return ParamType.ANY

    def _analyze_function_call(self, node: FunctionCallNode) -> ParamType:
        func_name = node.name

        if not self.context.is_operator(func_name):
            self.errors.append(f"Unknown operator: '{func_name}'")
            return ParamType.ANY

        spec = self.context.get_operator_spec(func_name)
        if not spec:
            self.errors.append(f"Unknown operator: '{func_name}'")
            return ParamType.ANY

        # 作用域校验（强约束：必须在 allowed_scopes=COMBO/REGULAR）
        self._check_operator_scope_allowed(func_name)

        # 参数数量校验
        self._check_param_count(node, spec)

        # 参数类型校验
        arg_types: List[ParamType] = []
        for i, arg in enumerate(node.args):
            arg_type = self._analyze_expression(arg)
            arg_types.append(arg_type)
            if i < len(spec.positional_params):
                param_spec = spec.positional_params[i]
                self._check_param_type(arg, arg_type, param_spec, func_name, i)

        for key, value_node in node.kwargs.items():
            if key not in spec.keyword_params:
                self.errors.append(f"Unknown keyword argument '{key}' for operator '{func_name}'")
                continue

            param_spec = spec.keyword_params[key]
            value_type = self._analyze_expression(value_node)
            self._check_param_type(value_node, value_type, param_spec, func_name, key)

            # 若是字面量，做 value_constraint 校验
            if isinstance(value_node, (NumberNode, StringNode, BoolNode)):
                is_valid, msg = param_spec.validate_value(value_node.value)
                if not is_valid:
                    self.errors.append(f"{func_name}: {msg}")

        # 缺失必填 keyword
        for key, param_spec in spec.keyword_params.items():
            if not param_spec.optional and param_spec.default_value is None:
                if key not in node.kwargs:
                    self.errors.append(f"{func_name}: Missing required keyword argument '{key}'")

        # 特殊规则（目前只加必要的，避免过度假阴性）
        self._check_special_rules(node, spec)

        return spec.return_type

    def _check_param_count(self, node: FunctionCallNode, spec: OperatorSpec):
        func_name = node.name
        arg_count = len(node.args)

        if spec.variadic:
            if arg_count < spec.min_args:
                self.errors.append(f"{func_name}: Expected at least {spec.min_args} arguments, got {arg_count}")
            return

        required_positional = sum(1 for p in spec.positional_params if not p.optional)
        total_positional = len(spec.positional_params)
        if arg_count < required_positional:
            self.errors.append(f"{func_name}: Expected at least {required_positional} positional arguments, got {arg_count}")
        elif arg_count > total_positional:
            self.errors.append(f"{func_name}: Expected at most {total_positional} positional arguments, got {arg_count}")

    def _check_param_type(
        self,
        arg_node: ASTNode,
        arg_type: ParamType,
        param_spec: ParamSpec,
        func_name: str,
        param_index_or_key: Any,
    ):
        expected_type = param_spec.param_type
        if expected_type == ParamType.ANY:
            if arg_type in {ParamType.GROUP, ParamType.STRING, ParamType.VECTOR}:
                # 为避免假阳性，这里保留“ANY不允许这些类型”的保守策略
                self.errors.append(
                    f"{func_name}: Parameter '{param_index_or_key}' cannot accept {arg_type.value} type via ANY"
                )
            return

        if expected_type == ParamType.MATRIX and arg_type == ParamType.VECTOR:
            if not (isinstance(arg_node, FunctionCallNode) and arg_node.name.lower().startswith("vec_")):
                self.errors.append(
                    f"{func_name}: Parameter '{param_index_or_key}' requires MATRIX type, but got VECTOR. "
                    f"Convert VECTOR using vec_* first."
                )
            return

        if expected_type == ParamType.GROUP and arg_type != ParamType.GROUP:
            self.errors.append(
                f"{func_name}: Parameter '{param_spec.name}' requires GROUP type, got {arg_type.value}"
            )
            return

        if expected_type != arg_type:
            if expected_type == ParamType.FLOAT and arg_type == ParamType.INT:
                return
            if arg_type != ParamType.ANY:
                self.errors.append(
                    f"{func_name}: Parameter '{param_index_or_key}' type mismatch - expected {expected_type.value}, got {arg_type.value}"
                )

    def _check_special_rules(self, node: FunctionCallNode, spec: OperatorSpec):
        # ts_backfill 特殊：要求要么第二位置参数 d，要么 lookback 关键字参数；并且二者不能同时出现（保守）
        if node.name.lower() == "ts_backfill":
            has_positional_lookback = len(node.args) >= 2
            has_keyword_lookback = "lookback" in node.kwargs

            if has_positional_lookback and has_keyword_lookback:
                self.errors.append("ts_backfill: Cannot use both positional lookback and keyword lookback.")
            elif not has_positional_lookback and not has_keyword_lookback:
                self.errors.append("ts_backfill: Must provide either positional lookback (d) or keyword lookback.")

            # ignore 在 operators.json 里允许，因此不再禁止 ignore（与你的文字一致）

    def _map_infix_op_to_operator(self, op: str) -> Optional[str]:
        return {
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
        }.get(op)

    def _map_unary_op_to_operator(self, op: str) -> Optional[str]:
        # unary '-' 等价于 reverse(x)
        if op == "-":
            return "reverse"
        return None

    def _check_unused_variables(self):
        unused = set(self.defined_vars.keys()) - self.used_vars
        for var in unused:
            self.errors.append(f"Variable '{var}' is defined but never used")


# ============================================================================
# 第七部分：对外 API
# ============================================================================


def validate_expression(
    expression: str,
    csv_path: Optional[str] = None,
    operators_json_path: str = "operators.json",
    allowed_scopes: Tuple[str, ...] = ("COMBO", "REGULAR"),
    required_root_scope: str = "REGULAR",
    allow_unknown_fields: bool = True,
) -> Tuple[bool, List[str]]:
    try:
        tokenizer = Tokenizer(expression)
        tokens = tokenizer.tokenize()

        parser = Parser(tokens)
        ast = parser.parse()

        context = DataContext(
            csv_path=csv_path,
            operators_json_path=operators_json_path,
        )
        analyzer = SemanticAnalyzer(
            context=context,
            allowed_scopes=set(allowed_scopes),
            required_root_scope=required_root_scope,
            allow_unknown_fields=allow_unknown_fields,
        )
        return analyzer.analyze(ast)
    except SyntaxError as e:
        return False, [f"Syntax error: {str(e)}"]
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]


def validate_expression_detailed(
    expression: str,
    csv_path: Optional[str] = None,
    operators_json_path: str = "operators.json",
    allowed_scopes: Tuple[str, ...] = ("COMBO", "REGULAR"),
    required_root_scope: str = "REGULAR",
    allow_unknown_fields: bool = True,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "expression": expression,
        "is_valid": False,
        "errors": [],
        "token_count": 0,
        "statement_count": 0,
        "required_root_scope": required_root_scope,
        "allowed_scopes": list(allowed_scopes),
    }
    try:
        tokenizer = Tokenizer(expression)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()

        context = DataContext(
            csv_path=csv_path,
            operators_json_path=operators_json_path,
        )
        analyzer = SemanticAnalyzer(
            context=context,
            allowed_scopes=set(allowed_scopes),
            required_root_scope=required_root_scope,
            allow_unknown_fields=allow_unknown_fields,
        )
        is_valid, errors = analyzer.analyze(ast)
        report["is_valid"] = is_valid
        report["errors"] = errors
        report["token_count"] = max(len(tokens) - 1, 0)
        report["statement_count"] = len(ast.statements)
        report["undefined_allowed"] = allow_unknown_fields
        report["field_catalog"] = context.field_catalog_status()
        if context.field_catalog_degraded:
            report["warnings"] = [
                "Field catalog is missing or empty; field-name and field-type validation is degraded."
            ]
        return report
    except SyntaxError as e:
        report["errors"] = [f"Syntax error: {str(e)}"]
        return report
    except Exception as e:
        report["errors"] = [f"Validation error: {str(e)}"]
        return report


def validate_expression_batch(
    expressions: List[str],
    csv_path: str = "USA_TOP1000_1.csv",
    operators_json_path: str = "operators.json",
    allowed_scopes: Tuple[str, ...] = ("COMBO", "REGULAR"),
    required_root_scope: str = "REGULAR",
    allow_unknown_fields: bool = True,
) -> List[Tuple[bool, List[str]]]:
    context = DataContext(csv_path=csv_path, operators_json_path=operators_json_path)
    results: List[Tuple[bool, List[str]]] = []

    for expr in expressions:
        try:
            tokenizer = Tokenizer(expr)
            tokens = tokenizer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()

            analyzer = SemanticAnalyzer(
                context=context,
                allowed_scopes=set(allowed_scopes),
                required_root_scope=required_root_scope,
                allow_unknown_fields=allow_unknown_fields,
            )
            results.append(analyzer.analyze(ast))
        except SyntaxError as e:
            results.append((False, [f"Syntax error: {str(e)}"]))
        except Exception as e:
            results.append((False, [f"Validation error: {str(e)}"]))

    return results


def _demo_main():
    print("Alpha表达式验证器（Scope 强约束：最终 REGULAR；算子仅 COMBO/REGULAR）")
    print("-" * 80)
    expression = 'bucket(returns,range="0,1,0.1")'
    is_valid, errors = validate_expression(expression, csv_path="EUR_TOP2500_1.csv")
    print("Expression:", expression)
    if is_valid:
        print("✓ VALID")
    else:
        print("✗ INVALID")
        for err in errors:
            print(" -", err)


if __name__ == "__main__":
    _demo_main()

