"""Report CONFIG.<field> usage across the repo.

Counts occurrences in Python files (excluding config.py) and flags fields used
once or never to keep config lean.

Use --apply to remove unused config fields and move single-use fields into the
only file that references them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import argparse
import ast
import inspect
import re
from pathlib import Path
import textwrap
from typing import Dict, Iterable

from util.config import GlobalConfig


CONFIG_REF = re.compile(r"\bCONFIG\.(\w+)\b")


@dataclass(frozen=True)
class FieldInfo:
    name: str
    start_line: int
    end_line: int
    expr: str | None


def _config_fields() -> set[str]:
    names = {field.name for field in fields(GlobalConfig) if not field.name.startswith("_")}
    for name, value in inspect.getmembers(GlobalConfig):
        if name.startswith("_"):
            continue
        if isinstance(value, property):
            names.add(name)
    return names


_SKIP_DIRS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "dist",
    "build",
    ".git",
}


def _property_fields() -> set[str]:
    return {
        name
        for name, value in inspect.getmembers(GlobalConfig)
        if not name.startswith("_") and isinstance(value, property)
    }


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.name == "config.py":
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def _load_config_source(config_path: Path) -> tuple[str, ast.Module]:
    source = config_path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _extract_expr(source: str, node: ast.AST | None) -> str | None:
    if node is None:
        return None
    expr = ast.get_source_segment(source, node)
    if expr is None:
        return None
    return textwrap.dedent(expr).strip()


def _field_expr(source: str, node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "field":
        for kw in node.keywords:
            if kw.arg == "default":
                return _extract_expr(source, kw.value)
            if kw.arg == "default_factory" and isinstance(kw.value, ast.Lambda):
                return _extract_expr(source, kw.value.body)
        return None
    return _extract_expr(source, node)


def _collect_field_infos(config_path: Path) -> dict[str, FieldInfo]:
    source, module = _load_config_source(config_path)
    field_infos: dict[str, FieldInfo] = {}

    class_node = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "GlobalConfig":
            class_node = node
            break
    if class_node is None:
        return field_infos

    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.startswith("_"):
                continue
            expr = _field_expr(source, node.value)
            field_infos[name] = FieldInfo(
                name=name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                expr=expr,
            )
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name.startswith("_"):
                continue
            expr = _field_expr(source, node.value)
            field_infos[name] = FieldInfo(
                name=name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                expr=expr,
            )

    return field_infos


def _collect_internal_refs(config_path: Path) -> set[str]:
    source, module = _load_config_source(config_path)

    class SelfRefVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                self.names.add(node.attr)
            self.generic_visit(node)

    visitor = SelfRefVisitor()
    visitor.visit(module)
    return visitor.names


def _usage_by_file(root: Path) -> dict[str, set[Path]]:
    usage: dict[str, set[Path]] = {}
    for path in _iter_python_files(root):
        text = path.read_text(encoding="utf-8")
        for match in CONFIG_REF.findall(text):
            usage.setdefault(match, set()).add(path)
    return usage


def audit_config_usage(root: Path) -> tuple[Dict[str, int], Dict[str, int]]:
    known = _config_fields()
    counts = {name: 0 for name in sorted(known)}
    unknown: Dict[str, int] = {}

    for path in _iter_python_files(root):
        text = path.read_text(encoding="utf-8")
        for match in CONFIG_REF.findall(text):
            if match in counts:
                counts[match] += 1
            else:
                unknown[match] = unknown.get(match, 0) + 1

    return counts, unknown


def _find_insertion_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if "CONFIG = load_config" in line:
            return idx + 1
    for idx, line in enumerate(lines):
        if "from util.config import load_config" in line:
            return idx + 1
    last_import = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import = idx + 1
    return last_import


def _build_block(constants: dict[str, str]) -> list[str]:
    block: list[str] = []
    for name in sorted(constants):
        expr = constants[name]
        lines = expr.splitlines() if expr else [""]
        if not lines:
            continue
        block.append(f"{name} = {lines[0]}")
        block.extend(lines[1:])
        block.append("")
    if block and block[-1] == "":
        block.pop()
    return block


def _apply_changes(root: Path) -> None:
    config_path = root / "util" / "config.py"
    field_infos = _collect_field_infos(config_path)
    internal_refs = _collect_internal_refs(config_path)
    usage_files = _usage_by_file(root)

    removals: list[str] = []
    moves: dict[Path, dict[str, str]] = {}
    skipped: list[str] = []

    for name, info in field_infos.items():
        files = usage_files.get(name, set())
        if name in internal_refs:
            continue
        if not files:
            removals.append(name)
            continue
        if len(files) == 1:
            if info.expr is None:
                skipped.append(name)
                continue
            target = next(iter(files))
            target_text = target.read_text(encoding="utf-8")
            if re.search(rf"^\s*{re.escape(name)}\s*=", target_text, re.MULTILINE):
                skipped.append(name)
                continue
            moves.setdefault(target, {})[name] = info.expr
            removals.append(name)

    for path, constants in moves.items():
        text = path.read_text(encoding="utf-8")
        for name in constants:
            text = re.sub(rf"\bCONFIG\.{re.escape(name)}\b", name, text)
        lines = text.splitlines()
        insert_at = _find_insertion_index(lines)
        block = _build_block(constants)
        if block:
            lines = lines[:insert_at] + [""] + block + [""] + lines[insert_at:]
        updated = "\n".join(lines)
        if text.endswith("\n"):
            updated += "\n"
        path.write_text(updated, encoding="utf-8")

    if removals:
        source = config_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        ranges = [
            (field_infos[name].start_line, field_infos[name].end_line)
            for name in removals
            if name in field_infos
        ]
        for start, end in sorted(ranges, reverse=True):
            del lines[start - 1 : end]
        updated = "\n".join(lines)
        if source.endswith("\n"):
            updated += "\n"
        config_path.write_text(updated, encoding="utf-8")

    print("Applied changes:")
    print("removed:", ", ".join(sorted(removals)) if removals else "none")
    if moves:
        print("moved:")
        for path, constants in sorted(moves.items(), key=lambda item: str(item[0])):
            print(f"- {path.relative_to(root)}: {', '.join(sorted(constants))}")
    else:
        print("moved: none")
    if skipped:
        print("skipped (no literal/default or already defined):", ", ".join(sorted(skipped)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move single-use fields into their only file and remove unused fields.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    counts, unknown = audit_config_usage(root)
    internal_refs = _collect_internal_refs(root / "util" / "config.py")
    for name in internal_refs:
        if name in counts and counts[name] < 2:
            counts[name] = 2
    property_fields = _property_fields()

    unused = [
        name
        for name, count in counts.items()
        if count == 0 and name not in property_fields
    ]
    single_use = [
        name
        for name, count in counts.items()
        if count == 1 and name not in property_fields
    ]

    print("Config usage counts (excluding config.py):")
    for name, count in sorted(counts.items(), key=lambda item: (item[1], item[0])):
        print(f"{count:>3}  {name}")

    if unknown:
        print("\nUnknown CONFIG attributes referenced:")
        for name, count in sorted(unknown.items(), key=lambda item: (item[1], item[0])):
            print(f"{count:>3}  {name}")

    print("\nFlags:")
    if unused:
        print("unused:", ", ".join(unused))
    else:
        print("unused: none")
    if single_use:
        print("single-use:", ", ".join(single_use))
    else:
        print("single-use: none")

    if args.apply:
        _apply_changes(root)


if __name__ == "__main__":
    main()
