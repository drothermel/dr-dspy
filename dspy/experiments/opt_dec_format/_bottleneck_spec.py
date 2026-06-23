"""Lazy access to dr-bottleneck's public workflow job schemas."""

from __future__ import annotations

import sys
import types
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def workflow_job_spec() -> ModuleType:
    """Import dr_bottleneck.workflow_jobs.spec from install or sibling checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    providers_src = (repo_root / "../dr-providers/src").resolve()
    if providers_src.exists() and str(providers_src) not in sys.path:
        sys.path.insert(0, str(providers_src))
    spec_path = (repo_root / "../dr-bottleneck/src/dr_bottleneck/workflow_jobs/spec.py").resolve()
    if spec_path.exists():
        return _load_spec_file(spec_path)
    return import_module("dr_bottleneck.workflow_jobs.spec")


def _load_spec_file(path: Path) -> ModuleType:
    module_name = "dr_bottleneck.workflow_jobs.spec"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.modules.setdefault("dr_bottleneck", types.ModuleType("dr_bottleneck"))
    sys.modules.setdefault(
        "dr_bottleneck.workflow_jobs",
        types.ModuleType("dr_bottleneck.workflow_jobs"),
    )
    module_spec = spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        msg = f"Cannot load dr-bottleneck workflow spec from {path}"
        raise ModuleNotFoundError(msg)
    module = module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module
