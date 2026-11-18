# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task discovery and optional user-created partitions; no bundled task lists."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from .. import config

_LB_DOMAINS = frozenset(
    {
        "calendar-management",
        "data-analysis",
        "document-processing",
        "financial-reasoning",
        "image-processing",
        "information-retrieval",
        "knowledge-qa",
        "sql-query",
        "finance",
        "manufacturing",
        "supply-chain",
    }
)

# The general suite labels some tasks in Chinese; one practice area gets one slug.
_LB_DOMAIN_ALIASES = {
    "信息检索": "information-retrieval",
    "知识问答": "knowledge-qa",
    "财务数值推理": "financial-reasoning",
    "sql数据查询": "sql-query",
    "supply_chain": "supply-chain",
}

_LB_ID_DOMAINS = frozenset({"finance", "manufacturing", "supply-chain"})
_LB_ID_DOMAIN = re.compile(
    r"^(?P<domain>" + "|".join(sorted(_LB_ID_DOMAINS, key=len, reverse=True)) + r")-"
)


def discover(bench: str, subset: str | None = None) -> dict[str, str]:
    """Return sorted task IDs mapped to their dataset-provided domain."""
    if bench == "jb":
        root = config.jb_repo() / "dataset" / "main"
        tasks = {
            p.parent.parent.relative_to(root).as_posix(): p.parent.parent.relative_to(root).parts[0]
            for p in sorted(root.glob("*/*/task_folder/TASK_INSTRUCTIONS.txt"))
        }
    elif bench == "lab":
        root = config.lab_repo() / "tasks"
        tasks = {
            p.parent.relative_to(root).as_posix(): p.parent.relative_to(root).parts[0]
            for p in sorted(root.glob("*/*/task.json"))
        }
    elif bench == "wb" and subset in ("office", "web", "code"):
        root = config.wb_repo() / "datasets" / f"wb-bench-{subset}-v1.0" / "tasks"
        tasks = {}
        field = "role" if subset == "code" else "category"
        for p in sorted(root.glob("*/task.toml")):
            meta = tomllib.loads(p.read_text(encoding="utf-8")).get("metadata", {})
            domain = meta.get(field)
            if not isinstance(domain, str) or not domain.strip():
                raise ValueError(f"{p}: metadata.{field} must be a nonempty string")
            if not (p.parent / "instruction.md").is_file():
                raise FileNotFoundError(f"Missing task instructions: {p.parent / 'instruction.md'}")
            tasks[p.parent.name] = domain.strip()
    elif bench == "lb" and subset in ("general", "enterprise"):
        root = config.lb_datasets() / f"lg-bench-{subset}-v1.0" / "tasks"
        tasks = {}
        for p in sorted(root.glob("*/task.toml")):
            meta = tomllib.loads(p.read_text(encoding="utf-8")).get("metadata", {})
            domain = lb_domain(p.parent.name, meta)
            if not (p.parent / "instruction.md").is_file():
                raise FileNotFoundError(f"Missing task instructions: {p.parent / 'instruction.md'}")
            tasks[p.parent.name] = domain
    else:
        raise ValueError(f"Unknown benchmark/subset: {bench}/{subset}")
    if not tasks:
        raise FileNotFoundError(
            f"No {bench} tasks found at {root}. Run harness2-setup "
            f"{bench}" + (f" --subset {subset}" if subset else "") + "."
        )
    return tasks


def lb_domain(task_id: str, meta: dict) -> str:
    """The lingee-bench grouping unit: metadata.domain, normalised to one slug per area.

    The general suite ships domain labels in two languages and mixed case, and the
    enterprise suite ships none at all, so both are folded onto the same small set of
    practice areas. A label outside that set raises instead of silently becoming its own
    domain, because a single-task library is not a domain.
    """
    raw = meta.get("domain")
    if isinstance(raw, str) and raw.strip():
        key = _LB_DOMAIN_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        slug = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
        if slug in _LB_DOMAINS:
            return slug
        raise ValueError(
            f"lingee-bench task {task_id}: metadata.domain {raw!r} is not a known practice "
            f"area {sorted(_LB_DOMAINS)}; add it to tasks._LB_DOMAIN_ALIASES"
        )
    m = _LB_ID_DOMAIN.match(task_id)
    if m:
        return m.group("domain")
    raise ValueError(
        f"lingee-bench task {task_id}: no metadata.domain and no labelled task-id prefix; "
        f"expected one of {sorted(_LB_ID_DOMAINS)}"
    )


def select(tasks: dict[str, str], split: str, path: Path | None, *, benchmark: str) -> list[str]:
    if split not in ("all", "train", "test"):
        raise ValueError(f"Unknown split {split!r}; use all, train, or test")
    if path is None:
        if split != "all":
            raise ValueError(
                "--split train/test requires --split-file; create one with "
                "python scripts/split_tasks.py, or use --split all"
            )
        return sorted(tasks)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("benchmark") != benchmark:
        raise ValueError(f"{path}: expected a split for {benchmark}")
    halves = []
    for name in ("train", "test"):
        ids = data.get(name)
        if not isinstance(ids, list) or any(not isinstance(t, str) for t in ids):
            raise ValueError(f"{path}: {name} must be a list of task IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate task IDs in {name}")
        halves.append(set(ids))
    if halves[0] & halves[1]:
        raise ValueError(f"{path}: train and test overlap")
    unknown = (halves[0] | halves[1]) - tasks.keys()
    if unknown:
        raise ValueError(f"{path}: task IDs absent from the dataset: {sorted(unknown)[:5]}")
    return sorted(halves[0] | halves[1] if split == "all" else data[split])
