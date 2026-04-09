from pathlib import Path
import ast
import sys
import textwrap

import mkdocs_gen_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.config import load_config  # noqa: E402

CONFIG = load_config("global")

api_module_filename = "main.py"
api_module = CONFIG.project_root / api_module_filename

docs_dir_name = "docs"

docs_gen_ref_filename = "gen_ref.py"

docs_script_skip_files = ("__init__.py",)

docs_scripts_dir_name = "scripts"

docs_util_dir_name = "util"

docs_util_skip_files = ("__init__.py", "config.py")

readme_filename = "README.md"


def _strip_module_docstring(source: str) -> tuple[str, str | None]:
    docstring = None
    doc_start = doc_end = None
    try:
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            doc_start = tree.body[0].lineno
            doc_end = getattr(tree.body[0], "end_lineno", None)
    except SyntaxError:
        docstring = None

    if doc_start and doc_end:
        lines = source.splitlines(keepends=True)
        source = "".join(lines[: doc_start - 1] + lines[doc_end:])
    return source, docstring


def _split_member_names(source: str) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    public_members = []
    internal_members = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        if name.startswith("_"):
            internal_members.append(name)
        else:
            public_members.append(name)
    return public_members, internal_members


def _write_module_docs(
    module_path: Path,
    out_dir: str,
    module_ref: str,
    *,
    title: str | None = None,
    doc_name: str | None = None,
) -> None:
    name = module_path.stem
    resolved_title = title or name.replace("_", " ").title()
    doc_path = Path(out_dir) / f"{doc_name or name}.md"
    source = module_path.read_text(encoding="utf-8")
    public_members, internal_members = _split_member_names(source)
    source, _ = _strip_module_docstring(source)
    with mkdocs_gen_files.open(doc_path, "w") as handle:
        handle.write(f"# {resolved_title}\n\n")
        handle.write(f"This page is generated from the `{module_ref}` module docstrings.\n\n")
        handle.write("## Public API\n\n")
        if public_members:
            handle.write(f"::: {module_ref}\n")
            handle.write("    options:\n")
            handle.write("      members:\n")
            for name in public_members:
                handle.write(f"        - {name}\n")
            handle.write("\n")
        else:
            handle.write("_No public API documented._\n\n")
        if internal_members:
            handle.write("## Internal API\n\n")
            for name in internal_members:
                handle.write(f"::: {module_ref}.{name}\n\n")
        handle.write("<details>\n")
        handle.write("<summary>View full source</summary>\n\n")
        handle.write("```python\n")
        handle.write(source)
        if not source.endswith("\n"):
            handle.write("\n")
        handle.write("```\n")
        handle.write("</details>\n")


def _write_script_source(module_path: Path, out_dir: str) -> None:
    name = module_path.stem
    title = name.replace("_", " ").title()
    doc_path = Path(out_dir) / f"{name}.md"
    source = module_path.read_text(encoding="utf-8")
    source, docstring = _strip_module_docstring(source)
    with mkdocs_gen_files.open(doc_path, "w") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(f"This page mirrors the script source at `scripts/{module_path.name}`.\n\n")
        if docstring:
            handle.write(docstring)
            handle.write("\n\n")
        handle.write("```python\n")
        handle.write(source)
        if not source.endswith("\n"):
            handle.write("\n")
        handle.write("```\n")


def _write_api_docs(module_path: Path, out_dir: str) -> None:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    doc_path = Path(out_dir) / "main.md"

    endpoints = []
    internals = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        display_name = f"`{name}`"
        doc = ast.get_docstring(node) or "TODO: add docstring."
        doc = textwrap.dedent(doc).strip()

        endpoint = None
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            if not isinstance(deco.func, ast.Attribute):
                continue
            if not isinstance(deco.func.value, ast.Name):
                continue
            if deco.func.value.id != "app":
                continue
            method = deco.func.attr.upper()
            if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                continue
            if deco.args and isinstance(deco.args[0], ast.Constant):
                route = str(deco.args[0].value)
                endpoint = f"{method} {route}"
                break

        segment = ast.get_source_segment(source, node) or f"def {name}(...):"

        if endpoint:
            endpoints.append((display_name, endpoint, doc, segment))
        elif name.startswith("_"):
            internals.append((display_name, endpoint, doc, segment))

    with mkdocs_gen_files.open(doc_path, "w") as handle:
        handle.write("# Material API Docs\n\n")
        handle.write("This page is generated from `main.py` and includes endpoint metadata when available.\n\n")
        if endpoints:
            handle.write("## Endpoints\n\n")
            for name, endpoint, doc, segment in endpoints:
                handle.write(f"### {name}\n\n")
                handle.write(f"**Endpoint:** `{endpoint}`\n\n")
                handle.write(f"{doc}\n\n")
                handle.write("<details>\n")
                handle.write("<summary>View endpoint code</summary>\n\n")
                handle.write("```python\n")
                handle.write(segment)
                if not segment.endswith("\n"):
                    handle.write("\n")
                handle.write("```\n")
                handle.write("</details>\n\n")
        if internals:
            handle.write("## Internal Functions\n\n")
            for name, _, doc, segment in internals:
                handle.write(f"### {name}\n\n")
                handle.write(f"{doc}\n\n")
                handle.write("<details>\n")
                handle.write("<summary>View function code</summary>\n\n")
                handle.write("```python\n")
                handle.write(segment)
                if not segment.endswith("\n"):
                    handle.write("\n")
                handle.write("```\n")
                handle.write("</details>\n\n")
        handle.write("<details>\n")
        handle.write("<summary>View full source</summary>\n\n")
        handle.write("```python\n")
        handle.write(source)
        if not source.endswith("\n"):
            handle.write("\n")
        handle.write("```\n")
        handle.write("</details>\n")


libraries = []
for module_path in sorted((CONFIG.project_root / docs_util_dir_name).glob("*.py")):
    if module_path.name in docs_util_skip_files:
        continue
    _write_module_docs(module_path, "libraries", f"util.{module_path.stem}")
    libraries.append(module_path.stem)

scripts = []
for module_path in sorted((CONFIG.project_root / docs_scripts_dir_name).glob("*.py")):
    if module_path.name in docs_script_skip_files:
        continue
    _write_script_source(module_path, "scripts")
    scripts.append(module_path.stem)

readme_source = (CONFIG.project_root / readme_filename).read_text(encoding="utf-8")
with mkdocs_gen_files.open("readme.md", "w") as handle:
    handle.write(readme_source)

if api_module.exists():
    _write_api_docs(api_module, "api")

about_source = (CONFIG.project_root / docs_dir_name / docs_gen_ref_filename).read_text(encoding="utf-8")
with mkdocs_gen_files.open("about_docs.md", "w") as handle:
    handle.write("# About These Docs\n\n")
    handle.write(
        "This documentation site is generated during the MkDocs build using "
        "`mkdocs-gen-files` and `mkdocstrings`.\n\n"
        "- `docs/gen_ref.py` scans `util/` and `scripts/` and writes reference pages.\n"
        "- Libraries pages render from module docstrings via `mkdocstrings`.\n"
        "- Script pages render the module docstring plus the full source code.\n\n"
        "## Generator Source\n\n"
    )
    handle.write("## Theming Notes\n\n")
    handle.write(
        "The docs theme is MkDocs Material. We override a few theme colors using\n"
        "`docs/styles.css`, which sets Material CSS variables based on tokens from\n"
        "`wherewild-design-system/src/theme.css`.\n\n"
        "If you want to tweak the palette, update `docs/styles.css` and rebuild the\n"
        "docs. You can also adjust theme options in `mkdocs.yml`.\n\n"
    )
    handle.write("```python\n")
    handle.write(about_source)
    if not about_source.endswith("\n"):
        handle.write("\n")
    handle.write("```\n")

with mkdocs_gen_files.open("SUMMARY.md", "w") as handle:
    handle.write("- Home\n")
    handle.write("    - [Overview](index.md)\n")
    handle.write("    - [README](readme.md)\n")
    handle.write("    - [About Docs](about_docs.md)\n")
    handle.write("- Libraries\n")
    handle.write("    - [Overview](libraries/index.md)\n")
    for name in libraries:
        title = name.replace("_", " ").title()
        handle.write(f"    - [{title}](libraries/{name}.md)\n")
    handle.write("- Scripts\n")
    handle.write("    - [Overview](scripts/index.md)\n")
    for name in scripts:
        title = name.replace("_", " ").title()
        handle.write(f"    - [{title}](scripts/{name}.md)\n")
    handle.write("- API\n")
    handle.write("    - [Overview](api/index.md)\n")
    if api_module.exists():
        handle.write("    - [Upload Processing](api/upload-processing.md)\n")
        handle.write("    - [Material API Docs](api/main.md)\n")
        handle.write("    - [Default FastAPI Docs](http://localhost:8000/docs)\n")
        handle.write("    - [Default OpenAPI Schema](http://localhost:8000/openapi.json)\n")
