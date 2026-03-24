"""Coverage tests for docs/gen_ref.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import uuid

from util import config as config_module


def _resolve_gen_ref_path() -> Path:
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        candidate = base / "docs" / "gen_ref.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Unable to locate docs/gen_ref.py from test path.")


GEN_REF_PATH = _resolve_gen_ref_path()


def _build_fake_project(root: Path, *, with_api: bool = True) -> None:
    (root / "util").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)

    (root / "util" / "__init__.py").write_text("", encoding="utf-8")
    (root / "util" / "config.py").write_text("IGNORED = True\n", encoding="utf-8")
    (root / "util" / "alpha.py").write_text(
        '"""Alpha docs."""\n\n'
        "def public_fn():\n"
        "    \"\"\"Public.\"\"\"\n"
        "    return 1\n\n"
        "class PublicClass:\n"
        "    pass\n\n"
        "def _internal_fn():\n"
        "    return 2\n",
        encoding="utf-8",
    )

    (root / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (root / "scripts" / "runner.py").write_text(
        '"""Runner docs."""\n\n'
        "def run():\n"
        "    return 42",
        encoding="utf-8",
    )

    if with_api:
        (root / "main.py").write_text(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n\n"
            "@app.get('/x')\n"
            "def endpoint():\n"
            "    \"\"\"Endpoint doc.\"\"\"\n"
            "    return {'ok': True}\n\n"
            "@app.trace('/ignored')\n"
            "def ignored_method():\n"
            "    return 1\n\n"
            "def _internal():\n"
            "    \"\"\"Internal doc.\"\"\"\n"
            "    return None",
            encoding="utf-8",
        )

    (root / "README.md").write_text("# Demo Readme", encoding="utf-8")
    # No trailing newline to hit newline append branch.
    (root / "docs" / "gen_ref.py").write_text("print('about docs source')", encoding="utf-8")


def _load_gen_ref(monkeypatch, project_root: Path, output_root: Path):
    def _open(rel_path: str | Path, mode: str = "w"):
        path = output_root / Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open(mode, encoding="utf-8")

    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", types.SimpleNamespace(open=_open))
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda _name: types.SimpleNamespace(project_root=project_root),
    )

    module_name = f"gen_ref_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, GEN_REF_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gen_ref_top_level_generation(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    output_root = tmp_path / "generated"
    _build_fake_project(project_root, with_api=True)
    repo_root = str(GEN_REF_PATH.resolve().parents[1])
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo_root])

    _load_gen_ref(monkeypatch, project_root, output_root)

    assert (output_root / "libraries" / "alpha.md").exists()
    assert (output_root / "scripts" / "runner.md").exists()
    assert (output_root / "api" / "main.md").exists()
    assert (output_root / "readme.md").read_text(encoding="utf-8").startswith("# Demo Readme")
    summary = (output_root / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Material API Docs" in summary
    assert "Default FastAPI Docs" in summary
    about = (output_root / "about_docs.md").read_text(encoding="utf-8")
    assert "Generator Source" in about
    assert "```python" in about


def test_gen_ref_helper_branches(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    output_root = tmp_path / "generated"
    _build_fake_project(project_root, with_api=False)
    module = _load_gen_ref(monkeypatch, project_root, output_root)

    # _strip_module_docstring success + syntax error fallback.
    source = '"""mod doc"""\n\ndef fn():\n    return 1\n'
    stripped, doc = module._strip_module_docstring(source)
    assert doc == "mod doc"
    assert '"""mod doc"""' not in stripped
    bad_source = "def broken(:\n"
    stripped_bad, doc_bad = module._strip_module_docstring(bad_source)
    assert stripped_bad == bad_source and doc_bad is None

    # _split_member_names success + syntax error branch.
    public, internal = module._split_member_names(
        "def a():\n    pass\n\n"
        "async def b():\n    pass\n\n"
        "class C:\n    pass\n\n"
        "def _d():\n    pass\n"
    )
    assert public == ["a", "b", "C"]
    assert internal == ["_d"]
    assert module._split_member_names("def bad(:") == ([], [])

    # _write_module_docs no-public branch + internal rendering.
    module_file = project_root / "util" / "only_internal.py"
    module_file.write_text("def _hidden():\n    return 1", encoding="utf-8")
    module._write_module_docs(
        module_file,
        "libraries",
        "util.only_internal",
        title="Only Internal",
        doc_name="only_internal_custom",
    )
    module_doc = (output_root / "libraries" / "only_internal_custom.md").read_text(encoding="utf-8")
    assert "_No public API documented._" in module_doc
    assert "## Internal API" in module_doc
    assert "::: util.only_internal._hidden" in module_doc

    # _write_script_source with/without module docstring.
    script_file = project_root / "scripts" / "tiny.py"
    script_file.write_text("def run():\n    return 1", encoding="utf-8")
    module._write_script_source(script_file, "scripts")
    script_doc = (output_root / "scripts" / "tiny.md").read_text(encoding="utf-8")
    assert "This page mirrors the script source" in script_doc
    assert "```python" in script_doc

    # _write_api_docs with endpoint + internal + missing docstring fallback.
    api_file = project_root / "api_test.py"
    api_file.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n\n"
        "def deco(fn):\n"
        "    return fn\n\n"
        "class Holder:\n"
        "    pass\n"
        "holder = Holder()\n"
        "other = Holder()\n\n"
        "@deco\n"
        "def plain_decorator():\n"
        "    return 0\n\n"
        "@deco()\n"
        "def call_name_decorator():\n"
        "    return 0\n\n"
        "@holder.app.get('/skip')\n"
        "def nested_attr_decorator():\n"
        "    return 0\n\n"
        "@other.get('/skip2')\n"
        "def wrong_root_decorator():\n"
        "    return 0\n\n"
        "@app.get('/ok')\n"
        "def ok():\n"
        "    return 1\n\n"
        "def _internal_fn():\n"
        "    return 2",
        encoding="utf-8",
    )
    module._write_api_docs(api_file, "api")
    api_doc = (output_root / "api" / "main.md").read_text(encoding="utf-8")
    assert "## Endpoints" in api_doc
    assert "**Endpoint:** `GET /ok`" in api_doc
    assert "TODO: add docstring." in api_doc
    assert "## Internal Functions" in api_doc
    assert "View full source" in api_doc
