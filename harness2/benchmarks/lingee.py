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

"""lingee-bench adapter: one (subset, substrate) cell per `LbAdapter` instance.

Execution is a Harbor batch through the lingee-bench runner: one label is one
rollout-store namespace, and one batch is one generated job YAML handed to
`uv run harbor run --config`. The editable harness reaches the container as a read-only
bind mount that the runner's agent installs into the agent's own OpenCode config, so the
benchmark keeps owning its container, its model and MCP proxies, and its graders.

The graders write `/logs/verifier/score.json` (general) or `results.json` plus
`reward.json` (enterprise) with an explicit status. A status outside the graded set
means the run is UNSCORED -- `score()` returns None, never a zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.selection import order_graded_first
from ..core.spec import BenchPrompts, BenchSpec, resolve_channel_model
from ..substrate import opencode as OC
from ..substrate.base import Install, RunRequest
from ..substrate.opencode import OPENCODE_SURFACE, OpencodeSubstrate
from .tasks import discover, select

SUBSTRATES = ("oc",)
SCOREABLE = frozenset({"general", "enterprise"})

TRAJ_MAX_CHARS = 2_000_000
DELIVERABLE_CHAR_CAP = 200_000
DELIVERABLE_FILE_CAP = 40
JUDGE_BASE_URL = "http://lg-model-proxy:18080/openai"
JUDGE_API_KEY = "lg-proxy-placeholder"
MCP_PROXY_URL = "http://lg-mcp-proxy:18090/mcp"
EPHEMERAL_PREFIX = "_lb-"
HARBOR_NAME_MAX = 32

_GRADED_TRIAL = re.compile(r"^(?P<task>.+)__[A-Za-z0-9]+$")

_label_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_slots = {s: threading.Semaphore(config.LB_MAX_JOBS) for s in SUBSTRATES}


@dataclass(frozen=True)
class Subset:
    """One lingee-bench subset: its dataset, domain unit, deliverable and scoring."""

    name: str
    dataset_id: str
    domain_word: str
    deliverable: str
    scoring: str
    needs_erp: bool
    leak_globs: tuple[str, ...]

    @property
    def tasks_dir(self) -> Path:
        return config.lb_datasets() / self.dataset_id / "tasks"


SUBSETS: dict[str, Subset] = {
    "general": Subset(
        name="general",
        dataset_id="lg-bench-general-v1.0",
        domain_word="practice area",
        deliverable="the files the request asks for, written to /workspace/output",
        scoring="dual_channel",
        needs_erp=False,
        leak_globs=("tests/**", "environment/**", "solution/**"),
    ),
    "enterprise": Subset(
        name="enterprise",
        dataset_id="lg-bench-enterprise-v1.0",
        domain_word="business domain",
        deliverable="the ERP deliverable (workbook, report or JSON) written to /workspace/output",
        scoring="erp_graded",
        needs_erp=True,
        leak_globs=("tests/**", "environment/**", "solution/**"),
    ),
}


def label_sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", s)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] lb | {msg}", flush=True)


def _join(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _norm(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (value or "").lower())).strip("_")


# --------------------------------------------------------------------------------------
# Benchmark-shaped text. BenchPrompts rejects any mention of the measurement, so these
# describe the work a practitioner does, never the artifacts that judge it.
# --------------------------------------------------------------------------------------

LINGEE_BASES: dict[str, BenchPrompts] = {
    "general": BenchPrompts(
        work=(
            "Delivering a business work product from a Chinese-language request: analysing a "
            "spreadsheet, reconciling documents, retrieving an answer from a reference set, "
            "reading an invoice image, or scheduling against a policy. The request names the "
            "outcome, not the method, and the delivered file is what the requester opens."
        ),
        domain_word="practice area",
        deliverable_word="work product",
        inv_levers=(
            "- find every supplied input first and read the request's vocabulary for the exact "
            "field, sheet, column and unit it names\n"
            "- establish the output path, container format and schema early, then fill it with "
            "code rather than by hand\n"
            "- separate missing, zero and not-applicable values; keep the requester's naming, "
            "ordering and date conventions\n"
            "- reopen the written file with an independent reader and check totals, row counts, "
            "formulas and required sections before answering"
        ),
        dom_levers=(
            "- encode reusable mechanics for the area: pivot and variance arithmetic, "
            "document extraction, evidence-anchored retrieval, OCR reconciliation, "
            "calendar arithmetic across policies\n"
            "- treat the reference material in the workspace as the only authority; quote the "
            "source location instead of asserting an external fact\n"
            "- keep the reasoning traceable: one line of evidence per claim, and a stated rule "
            "for every filter, join or denominator"
        ),
        yardstick=(
            "Did it read the request precisely, use every material input, compute or retrieve "
            "the right answer with a reviewable rule, write the asked-for file in the asked-for "
            "shape, and verify the result before finishing?"
        ),
        selection_checks=(
            "Prefer the work product that opens in its real format, contains every requested "
            "field, sheet and section, keeps the source's meaning and units, applies the stated "
            "rules consistently, and leaves no placeholder or invented figure behind."
        ),
    ),
    "enterprise": BenchPrompts(
        work=(
            "Executing an ERP back-office request end to end: looking up master and transaction "
            "data through the ERP tools, posting or checking a document, and delivering the "
            "workbook or report the requester will reconcile against the ledger."
        ),
        domain_word="business domain",
        deliverable_word="ERP work product",
        inv_levers=(
            "- resolve the organisation, period, account and item identifiers from the request "
            "before querying; an ambiguous lookup is answered by paging the data, not guessing\n"
            "- keep every figure traceable to the document it came from, and reconcile control "
            "totals (debit against credit, quantity against amount) before writing\n"
            "- write the asked-for workbook, sheet names and columns exactly, using code for the "
            "arithmetic\n"
            "- reopen the deliverable and re-check the numbers a reviewer would check first"
        ),
        dom_levers=(
            "- encode reusable mechanics per domain: voucher and balance arithmetic, cost "
            "roll-ups, settlement and reversal chains, inventory movements, capacity and "
            "delivery scheduling\n"
            "- respect the ERP's own accounting rules (period cut-off, currency, rounding, "
            "document state) instead of re-deriving them\n"
            "- state the query path behind each number so a reviewer can replay it"
        ),
        yardstick=(
            "Did it query the right entities and periods, apply the ERP's own rules, reconcile "
            "the totals, deliver the exact workbook asked for, and show where each figure came "
            "from?"
        ),
        selection_checks=(
            "Prefer the work product whose figures reconcile with the queried documents, whose "
            "workbook has the requested sheets and columns, which states its assumptions instead "
            "of hiding them, and which a finance reviewer could replay without asking a question."
        ),
    ),
}

LINGEE_DOMAINS: dict[str, str] = {
    "data-analysis": (
        "PRACTICE LENS — DATA ANALYSIS. Fix the grain, the period and the denominator first; "
        "reconcile row counts and control totals after every transformation; label every "
        "variance with the factor that explains it and name the rows that drive it."
    ),
    "document-processing": (
        "PRACTICE LENS — DOCUMENT PROCESSING. Extract with source locations, preserve field "
        "order and naming, and never summarise where the request asks for extraction; record "
        "deduplication and merge keys explicitly."
    ),
    "information-retrieval": (
        "PRACTICE LENS — RETRIEVAL. Answer from the supplied material only, cite the document "
        "and section behind each statement, and say plainly when the material does not contain "
        "the answer."
    ),
    "knowledge-qa": (
        "PRACTICE LENS — KNOWLEDGE ANSWERING. Resolve the question's scope and units, answer in "
        "the requested format, and anchor contested claims to the reference text."
    ),
    "sql-query": (
        "PRACTICE LENS — QUERY WORK. State the join keys, filters and aggregation grain before "
        "writing the query; check the result against a hand-computed sample and report the row "
        "count you expect."
    ),
    "financial-reasoning": (
        "PRACTICE LENS — FINANCIAL REASONING. Keep units and currency explicit, reconcile the "
        "components against the total, and show the arithmetic that links inputs to the figure "
        "reported."
    ),
    "image-processing": (
        "PRACTICE LENS — IMAGE RECONCILIATION. Read the documents field by field, keep what is "
        "legible separate from what is inferred, and reconcile recognised totals against the "
        "structured side."
    ),
    "calendar-management": (
        "PRACTICE LENS — SCHEDULING. Resolve time zones, working hours and policy precedence "
        "before proposing a slot; state the rule that moved each candidate."
    ),
    "finance": (
        "BUSINESS LENS — FINANCE. Honour period cut-off, currency and rounding rules; reconcile "
        "sub-ledger against general ledger before delivering; state the document state each "
        "figure depends on."
    ),
    "manufacturing": (
        "BUSINESS LENS — MANUFACTURING. Track quantities through order, issue and completion; "
        "reconcile scrap, yield and cost roll-ups; keep item and routing identifiers exact."
    ),
    "supply-chain": (
        "BUSINESS LENS — SUPPLY CHAIN. Fix the planning horizon and unit of measure; reconcile "
        "demand, supply and settlement documents; make lead-time and capacity assumptions "
        "explicit."
    ),
}


def lingee_prompts(subset: str, *, domain: str = "") -> BenchPrompts:
    """A lingee-bench subset profile plus the practice-area lens for one domain."""
    if subset not in LINGEE_BASES:
        raise ValueError(f"unknown lingee-bench subset {subset!r}; expected general or enterprise")
    base = LINGEE_BASES[subset]
    lens = LINGEE_DOMAINS.get(_norm(domain).replace("_", "-"), "")
    return base if not lens else replace(base, domain_guidance=_join(base.domain_guidance, lens))


def _task_universe(subset: str) -> tuple[str, ...]:
    return tuple(discover("lb", subset))


@lru_cache(maxsize=8)
def _exact_index(subset: str) -> dict[str, str]:
    idx: dict[str, str] = {}
    for tid in _task_universe(subset):
        idx.setdefault(tid, tid)
        idx.setdefault(tid[:HARBOR_NAME_MAX], tid)
    return idx


@lru_cache(maxsize=4096)
def resolve_task(name: str, subset: str) -> str | None:
    """Map a trial directory's task segment back to a task id (Harbor may truncate)."""
    hit = _exact_index(subset).get(name)
    if hit is not None:
        return hit
    cands = [t for t in _task_universe(subset) if t.startswith(name)]
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise RuntimeError(
            f"trial name {name!r} is ambiguous across {len(cands)} tasks: {cands[:3]}"
        )
    return None


# --------------------------------------------------------------------------------------
# Reading a finished trial: the graded status set is the contract.
# --------------------------------------------------------------------------------------

# General graders: dual_channel is the merged rule+judge number; core_threshold_failed is a
# real zero because the required artifact never appeared. Everything else (invalid_*,
# rule_layer_error, evaluation_error) is a grading failure, not a score.
_GENERAL_GRADED = frozenset({"dual_channel", "core_threshold_failed"})
_ERP_GRADED = frozenset({"graded"})


def _verifier_dir(run_dir: Path) -> Path:
    """Single-turn tasks read verifier/; multi-turn tasks read the last step's verifier/."""
    steps = sorted(p for p in (Path(run_dir) / "steps").glob("*/verifier") if p.is_dir())
    direct = Path(run_dir) / "verifier"
    if steps and not direct.is_dir():
        return steps[-1]
    return direct


def _json_or_none(p: Path) -> dict | None:
    try:
        got = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return got if isinstance(got, dict) else None


def _unit_float(v) -> float | None:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if 0.0 <= n <= 1.0 else None


def graded_score(run_dir: Path) -> tuple[str, float] | None:
    """(status, score) for a trial that finished grading, or None when it is unscored.

    Missing, torn or non-graded output is None -- the same value as "the grader failed" --
    because reporting it as a zero would turn an infrastructure fault into a model failure.
    """
    verifier = _verifier_dir(Path(run_dir))
    detail = _json_or_none(verifier / "score.json")
    if detail is not None:
        status = str(detail.get("status") or "")
        if status not in _GENERAL_GRADED:
            return None
        raw = _unit_float(detail.get("score"))
        if raw is None:
            raw = _unit_float((_json_or_none(verifier / "reward.json") or {}).get("reward"))
        return (status, raw) if raw is not None else None
    detail = _json_or_none(verifier / "results.json")
    if detail is not None:
        status = str(detail.get("status") or "")
        if status not in _ERP_GRADED:
            return None
        rewards = _json_or_none(verifier / "reward.json") or {}
        raw = (
            _unit_float(rewards.get("overall"))
            or _unit_float(rewards.get("reward"))
            or _unit_float(detail.get("content_score"))
        )
        return (status, raw) if raw is not None else None
    quick = _json_or_none(verifier / "reward.json")
    raw = _unit_float((quick or {}).get("overall")) or _unit_float((quick or {}).get("reward"))
    return ("reward.json", raw) if raw is not None else None


def _trial_finished(trial: Path) -> bool:
    """A terminal trial has a verifier directory; only score() reads the value in it."""
    return _verifier_dir(Path(trial)).is_dir()


def _trial_graded(trial: Path) -> bool:
    """Ranking predicate only: an ungraded trial ranks last, never drops out."""
    return graded_score(trial) is not None


def job_document(
    subset: Subset,
    task_id: str,
    *,
    jobs_dir: Path,
    fragments: list[Path],
    model: str,
    judge: str,
    agent_timeout: int,
    harness_ws: str | None,
    k: int = 1,
) -> str:
    """The generated job YAML: one task, k attempts, the benchmark's own runtime."""
    lines = [
        "# GENERATED by harness2/benchmarks/lingee.py -- ephemeral, safe to delete.",
        f"# subset={subset.name} task={task_id}",
        f"jobs_dir: {json.dumps(str(jobs_dir))}",
        f"n_attempts: {int(k)}",
        "n_concurrent_trials: 1",
        "retry:",
        "  max_retries: 0",
        "environment:",
        "  type: docker",
        "  extra_docker_compose:",
    ]
    lines += [f"    - {json.dumps(str(p))}" for p in fragments]
    if not subset.needs_erp and judge:
        lines += [
            "verifier:",
            "  env:",
            f"    JUDGE_BASE_URL: {json.dumps(JUDGE_BASE_URL)}",
            f"    JUDGE_API_KEY: {json.dumps(JUDGE_API_KEY)}",
            f"    JUDGE_MODEL: {json.dumps(judge)}",
            '    JUDGE_STRICT: "1"',
        ]
    lines += [
        "agents:",
        "  - import_path: lg_bench.agents.harness:LgHarnessOpenCode"
        if harness_ws
        else "  - import_path: lg_bench.agents.opencode:LgOpenCode",
        f"    model_name: {json.dumps(model)}",
        f"    override_timeout_sec: {int(agent_timeout)}",
        "    override_setup_timeout_sec: 1800",
    ]
    if harness_ws:
        lines += [
            "    kwargs:",
            "      install_mode: image",
            f"      version: {json.dumps(config.LB_OPENCODE_VER)}",
            f"      harness_ws: {json.dumps(harness_ws)}",
        ]
    else:
        lines += [
            "    kwargs:",
            "      install_mode: image",
            f"      version: {json.dumps(config.LB_OPENCODE_VER)}",
        ]
    if subset.needs_erp:
        lines += [
            "    mcp_servers:",
            "      - name: erp",
            "        transport: streamable-http",
            f"        url: {json.dumps(MCP_PROXY_URL)}",
        ]
    lines += [
        "tasks:",
        f"  - path: {json.dumps(str(subset.tasks_dir / task_id))}",
        "artifacts:",
        '  - "/workspace/output"',
    ]
    return "\n".join(lines) + "\n"


def mcp_allow_fragment(task_id: str, meta: dict) -> str:
    """The one MCP allowlist fragment for an enterprise task, matched by longest prefix.

    Fragments are named after the ERP case (`mcp-allow-gl-03.yaml`), while task ids carry a
    business-domain prefix and a difficulty suffix, so the case key is matched as a prefix
    rather than reconstructed.
    """
    compose = config.lb_repo() / "configs" / "compose"
    frags = sorted(p.stem.removeprefix("mcp-allow-") for p in compose.glob("mcp-allow-*.yaml"))
    if not frags:
        raise RuntimeError(
            f"no MCP allowlist fragments under {compose}; run harness2-setup lb --subset enterprise"
        )
    key = re.sub(r"-L\d$", "", task_id)
    key = re.sub(r"^(finance|manufacturing|supply-chain)-", "", key)
    case = str(meta.get("case_id") or "").strip().lower()
    hits = sorted({f for f in frags if key.startswith(f) or f.startswith(key) or (case and f.startswith(case))})
    if len(hits) != 1:
        raise RuntimeError(
            f"{task_id}: expected exactly one MCP allowlist fragment, matched {hits} from "
            f"{len(frags)} candidates; the dataset and the runner checkout disagree"
        )
    return hits[0]


def _open_permissions(root: Path) -> None:
    """Make a staged tree readable by the container user it is bind-mounted for."""
    for path in (root, *root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            continue
        extra = stat.S_IRGRP | stat.S_IROTH
        if stat.S_ISDIR(mode) or mode & stat.S_IXUSR:
            extra |= stat.S_IXGRP | stat.S_IXOTH
        path.chmod(mode | extra)


class LbAdapter:
    """One (subset, substrate) cell of lingee-bench."""

    def __init__(self, subset: str = "general", *, substrate: str = "oc", split_file: Path | None = None):
        if subset not in SUBSETS:
            raise ValueError(f"unknown subset {subset!r}; one of {sorted(SUBSETS)}")
        if substrate not in SUBSTRATES:
            raise ValueError(
                f"--bench lb runs the opencode substrate only, got {substrate!r}: the "
                f"lingee-bench runner ships OpenCode, Claude Code, Pi and DeepSeek harnesses, "
                f"and only the OpenCode one takes an editable harness tree"
            )
        self.subset = subset
        self.subset_info = SUBSETS[subset]
        self.split_file = split_file
        self.spec = spec_for(subset)
        self.substrate = OpencodeSubstrate(launcher=self._launch)
        self._deliverables: dict[str, dict[str, str]] = {}

    # -- discovery ------------------------------------------------------------------
    def tasks(self, split: str = "all") -> list[str]:
        return select(
            discover("lb", self.subset), split, self.split_file, benchmark=f"lb-{self.subset}"
        )

    def domain_of(self, task_id: str) -> str:
        return discover("lb", self.subset)[task_id]

    def reference_domain_order(self) -> list[str]:
        return sorted(set(discover("lb", self.subset).values()))

    def prompts_for(self, task_id: str) -> BenchPrompts:
        return lingee_prompts(self.subset, domain=self.domain_of(task_id))

    def task_prompt(self, task_id: str) -> str:
        """instruction.md verbatim: the ask as the requester wrote it, nothing rewritten."""
        return (self.subset_info.tasks_dir / task_id / "instruction.md").read_text(
            encoding="utf-8", errors="replace"
        )

    def task_files(self, task_id: str) -> Path | None:
        """The task directory; spec.leak_globs strips the answer key before any copy."""
        d = self.subset_info.tasks_dir / task_id
        return d if d.is_dir() else None

    # -- rollout plumbing -----------------------------------------------------------
    def _label(self, run_id: str, n: int = 1) -> str:
        label = run_id.replace("/", "__") + (f"__x{n}" if n > 1 else "")
        prefix = f"{self.spec.name}-{self.substrate.name}-"
        if not label.startswith(prefix):
            raise ValueError(
                f"run id {run_id!r} does not belong to this cell (expected a namespace "
                f"starting {prefix!r}); labels are constructed by core.schedule.ns only"
            )
        return label

    def _label_dir(self, label: str) -> Path:
        return self.spec.runs_root / "rollouts" / label_sanitize(label)

    def _trials(self, label: str, task_id: str) -> list[Path]:
        return _trials_in(_run_dirs(self._label_dir(label)), task_id, self.subset)

    def _rank(self, trials: list[Path]) -> list[Path]:
        newest = sorted(trials, key=lambda q: q.stat().st_mtime, reverse=True)
        return order_graded_first(newest, _trial_graded)

    def _cell_identity(self) -> dict:
        """What makes two rollouts of one label the same rollout, beyond the harness."""
        return {
            "solver": config.lb_solver_model(),
            "oc_ver": config.LB_OPENCODE_VER,
            "proxy_image": config.LB_PROXY_IMAGE,
        }

    def _identity_matches(self, trial: Path) -> bool:
        m = RM.read_mark(Path(trial)) or {}
        return all(m.get(key) == val for key, val in self._cell_identity().items())

    def _inputs(self, task_id: str) -> str:
        return RM.inputs_fingerprint(self.task_prompt(task_id), self.task_files(task_id))

    def _reusable(self, trial: Path, harness_dir: Path | None, task_id: str) -> bool:
        return (
            _trial_finished(trial)
            and RM.is_complete(trial, harness_dir=harness_dir, inputs=self._inputs(task_id))
            and self._identity_matches(trial)
        )

    def _install(self, harness_dir: Path | None) -> Install:
        """Stage the editable tree and move it somewhere the Docker daemon can bind-mount.

        `materialise` writes to the system temp directory, which is not a shared path on
        macOS; the staged copy is published under the cell's runs root instead.
        """
        source = Path(harness_dir) if harness_dir else self.base_harness()
        staged = self.substrate.materialise(source, self.spec, {})
        share = self.spec.runs_root / "_harness" / staged.fingerprint
        if not share.is_dir():
            share.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=".harness-", dir=share.parent))
            try:
                shutil.copytree(staged.staging_dir, tmp / "tree")
                os.replace(tmp / "tree", share)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        _open_permissions(share)
        shutil.rmtree(staged.staging_dir, ignore_errors=True)
        return Install(
            staging_dir=share, fingerprint=staged.fingerprint, extras={"install_mode": self.spec.install_mode}
        )

    def _fragments(self, task_id: str, install: Install, compose_dir: Path) -> list[Path]:
        """The compose overlays for one task, in the runner's own documented order."""
        repo = config.lb_repo()
        compose = repo / "configs" / "compose"
        out = [compose / "model-proxy.yaml", compose_dir / "model.yaml"]
        if self.subset_info.needs_erp:
            meta = _task_meta(self.subset_info.tasks_dir / task_id / "task.toml")
            allow = mcp_allow_fragment(task_id, meta)
            out += [
                compose / "mcp-proxy.yaml",
                compose / "erp-mock-mcp.yaml",
                compose / "erp-mock-benchmark.yaml",
                compose / f"mcp-allow-{allow}.yaml",
                compose / "erp-judge-env.yaml",
            ]
        else:
            out.append(compose / "general-isolation.yaml")
        out.append(compose / "harness" / "opencode.yaml")
        out.append(compose_dir / "harness.yaml")
        for path in out:
            if not path.is_file():
                raise FileNotFoundError(
                    f"compose fragment {path} is missing; run harness2-setup lb --subset {self.subset}"
                )
        return out

    def _write_fragments(self, compose_dir: Path, install: Install) -> None:
        compose_dir.mkdir(parents=True, exist_ok=True)
        allowed = [config.lb_solver_model()]
        judge = config.lb_judge_model()
        if judge and judge not in allowed:
            allowed.append(judge)
        (compose_dir / "model.yaml").write_text(
            "# GENERATED by harness2/benchmarks/lingee.py -- pin the proxy to this cell's models.\n"
            "services:\n"
            "  lg-model-proxy:\n"
            "    environment:\n"
            f"      LG_PROXY_ALLOWED_MODELS: {json.dumps(','.join(allowed))}\n",
            encoding="utf-8",
        )
        (compose_dir / "harness.yaml").write_text(
            "# GENERATED by harness2/benchmarks/lingee.py -- mount the editable harness tree.\n"
            "services:\n"
            "  main:\n"
            "    volumes:\n"
            "      - type: bind\n"
            f"        source: {json.dumps(str(install.staging_dir))}\n"
            f"        target: {json.dumps(config.LB_CONTAINER_HARNESS)}\n"
            "        read_only: true\n",
            encoding="utf-8",
        )

    def _job_env(self) -> dict[str, str]:
        """The runner's own credentials stay with it; only its LG_* names are inherited."""
        env = {
            **config.scrub_env(),
            **{k: v for k, v in os.environ.items() if k.startswith("LG_")},
            "LG_PROXY_ALLOWED_MODELS": ",".join(
                dict.fromkeys([config.lb_solver_model(), config.lb_judge_model()])
            ).strip(","),
        }
        judge = config.lb_judge_model()
        if judge:
            env.setdefault("LG_EVAL_JUDGE_MODEL", judge)
        return env

    def run_batch(
        self,
        label: str,
        tasks: list[str],
        install: Install,
        *,
        k: int,
        done,
        need=None,
        timeout_s: int | None = None,
    ) -> list[Path]:
        """Roll `tasks` k times each, returning the run dirs THIS invocation created."""
        wait = _label_lock(label)
        with wait:
            todo = [t for t in tasks if not done(t)]
            if not todo:
                _log(f"{label}: all {len(tasks)} task(s) already complete -- skipping")
                return []
            if len(todo) < len(tasks):
                _log(f"{label}: resuming, {len(tasks) - len(todo)} of {len(tasks)} already done")
            if need is not None:
                short = max(1, max(int(need(t)) for t in todo))
                if short != k:
                    _log(f"{label}: partial resume, asking for {short} attempt(s) not {k}")
                k = short

            task_id = todo[0]
            if len(todo) > 1:
                raise RuntimeError(
                    f"{label}: one Harbor job carries one task; got {len(todo)}: {todo[:3]}"
                )
            label_dir = self._label_dir(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            before = set(_run_dirs(label_dir))
            digest = hashlib.sha1(f"{label}|{task_id}".encode()).hexdigest()[:8]
            slug = f"{EPHEMERAL_PREFIX}{label_sanitize(label)[:40]}-{digest}"
            compose_dir = label_dir / "_compose" / slug
            self._write_fragments(compose_dir, install)
            meta = _task_meta(self.subset_info.tasks_dir / task_id / "task.toml")
            agent_timeout = int(float(meta.get("agent", {}).get("timeout_sec") or 1800))
            yaml_path = label_dir / f"{slug}.yaml"
            yaml_path.write_text(
                job_document(
                    self.subset_info,
                    task_id,
                    jobs_dir=label_dir,
                    fragments=self._fragments(task_id, install, compose_dir),
                    model=config.lb_solver_model(),
                    judge=config.lb_judge_model(),
                    agent_timeout=agent_timeout,
                    harness_ws=config.LB_CONTAINER_HARNESS,
                    k=k,
                ),
                encoding="utf-8",
            )
            wall = timeout_s or int(k * agent_timeout * 1.5 + 900)
            cmd = ["uv", "run", "harbor", "run", "--config", str(yaml_path), "--job-name", slug]
            if config.LB_ENV_FILE:
                cmd += ["--env-file", str(Path(config.LB_ENV_FILE).expanduser())]
            try:
                with _slots["oc"]:
                    _log(f"{label}: {task_id} x k={k} [oc]")
                    proc = subprocess.run(
                        cmd,
                        cwd=config.lb_repo(),
                        env=self._job_env(),
                        capture_output=True,
                        text=True,
                        timeout=wall,
                    )
                    if proc.returncode != 0:
                        _log(
                            f"{label}: harbor run exit {proc.returncode} "
                            f"(partial results are still usable) :: {(proc.stderr or '')[-400:]}"
                        )
            except subprocess.TimeoutExpired:
                _log(f"{label}: TIMEOUT after {wall}s -- the job group was killed")
            finally:
                yaml_path.unlink(missing_ok=True)
                shutil.rmtree(compose_dir, ignore_errors=True)
            return sorted(set(_run_dirs(label_dir)) - before)

    def _launch(self, req: RunRequest, install: Install) -> list[Path]:
        k = len(req.run_ids)
        label = self._label(req.run_ids[0], k)
        if self.substrate.name != "oc":
            raise RuntimeError("lingee-bench runs the oc substrate only")

        def _have(task_id: str) -> int:
            return sum(
                1
                for t in self._trials(label, task_id)
                if _trial_finished(t)
                and _mark_matches(t, install.fingerprint)
                and self._identity_matches(t)
            )

        new_dirs = self.run_batch(
            label,
            [req.task_ref],
            install,
            k=k,
            done=lambda tid: _have(tid) >= k,
            need=lambda tid: k - _have(tid),
            timeout_s=req.timeout or None,
        )
        fresh = [
            t for t in _trials_in(new_dirs, req.task_ref, self.subset) if _trial_finished(t)
        ]
        return self._rank(fresh)

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        """One rollout = a one-task Harbor job, idempotent by run id and harness id."""
        label = self._label(run_id)
        usable = [t for t in self._trials(label, task_id) if self._reusable(t, harness_dir, task_id)]
        if usable:
            return max(usable, key=lambda q: q.stat().st_mtime)
        install = self._install(harness_dir)
        # The published share is content-addressed by harness fingerprint and reused
        # across rollouts; it is never dropped here.
        fresh = self.substrate.execute(task_id, install, [run_id], int(timeout or 0))
        if not fresh:
            raise RuntimeError(
                f"Harbor produced no finished trial for {label} / {task_id}; see "
                f"{self._label_dir(label)} and the runner's own summary script"
            )
        chosen = fresh[0]
        RM.mark_complete(
            chosen,
            run_id=run_id,
            harness_dir=harness_dir,
            inputs=self._inputs(task_id),
            extra=self._cell_identity(),
        )
        return chosen

    def execute_many(
        self,
        task_id: str,
        harness_dir: Path | None,
        run_ids: list[str],
        *,
        timeout: int | None = None,
    ) -> list[Path]:
        """k replicates in ONE job. Only this invocation's trials are marked."""
        if not run_ids:
            return []
        k = len(run_ids)
        label = self._label(run_ids[0], k)

        def _mine() -> list[Path]:
            return [t for t in self._trials(label, task_id) if self._reusable(t, harness_dir, task_id)]

        mine = _mine()
        if len(mine) < k:
            install = self._install(harness_dir)
            fresh = self.substrate.execute(task_id, install, list(run_ids), int(timeout or 0))
            for t in fresh:
                if not RM.read_mark(t):
                    RM.mark_complete(
                        t,
                        run_id=run_ids[0],
                        harness_dir=harness_dir,
                        inputs=self._inputs(task_id),
                        extra=self._cell_identity(),
                    )
            mine = _mine()
        return self._rank(mine)[:k]

    # -- what the method reads back --------------------------------------------------
    def delivered(self, run_dir: Path) -> bool:
        """Did this rollout write anything into the declared artifact directory? Harbor
        writes its own manifest.json straight under artifacts/, so only artifact bodies
        at depth >= 1 (the collected /workspace/output tree) count."""
        root = Path(run_dir) / "artifacts"
        return any(
            p.is_file() and p.stat().st_size > 0 and p.parent != root for p in root.rglob("*")
        )

    def trajectory_path(self, run_dir: Path) -> Path:
        stream = Path(run_dir) / "agent" / "opencode.txt"
        return stream if stream.is_file() else Path(run_dir) / "agent" / "trajectory.json"

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        """The capture stream in the shared [ACTION]/[OBSERVATION] grammar."""
        f = self.trajectory_path(run_dir)
        if not f.is_file():
            return "(no trajectory)"
        return self.substrate.read_trajectory(f, obs_trunc=obs_trunc, max_chars=TRAJ_MAX_CHARS)

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        """filename -> readable text, one entry per artifact the rollout produced."""
        key = str(Path(run_dir).resolve())
        if key in self._deliverables:
            return dict(self._deliverables[key])
        out: dict[str, str] = {}
        root = Path(run_dir) / "artifacts"
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.parent != root):
            if len(out) >= int(DELIVERABLE_FILE_CAP):
                out["(truncated)"] = f"{int(DELIVERABLE_FILE_CAP)} artifacts shown"
                break
            out[path.relative_to(root).as_posix()] = _readable(path)
        self._deliverables[key] = dict(out)
        return out

    def deliverable_text(self, run_dir: Path) -> str:
        files = self.deliverable_files(run_dir)
        if not files:
            return ""
        return "\n\n".join(f"### {name}\n{body}" for name, body in files.items())

    def _killed(self, run_dir: Path) -> bool:
        """True when the capture stream ends on a provider error object."""
        try:
            raw = self.trajectory_path(run_dir).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        for ln in reversed(raw.splitlines()):
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                return False
            if not isinstance(ev, dict):
                return False
            return ev.get("type") == "error" or isinstance(ev.get("error"), dict)
        return False

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        """Read the grader's own verdict; ungraded or absent is None, never 0."""
        run_dir = Path(run_dir)
        if not self.delivered(run_dir) and self._killed(run_dir):
            return None
        got = graded_score(run_dir)
        if got is None:
            return None
        status, raw = got
        return {
            "n_passed": round(raw * 100),
            "n_criteria": 100,
            "all_pass": raw >= 1.0,
            "raw_score": raw,
            "verifier_status": status,
        }

    # -- harness -------------------------------------------------------------------
    def base_harness(self) -> Path:
        """H0: the stock editable tree the improvement pass starts from."""
        d = self.spec.runs_root / f"_base_harness_{self.substrate.name}_{os.getpid()}"
        if not HT.is_tree(OPENCODE_SURFACE, d):
            HT.write_tree(
                OPENCODE_SURFACE,
                {
                    "systemprompt.md": "",
                    "guardrails.md": "",
                    "LongTermMEMORY.md": "",
                    "tool_descriptions/tool_guidance.md": "",
                },
                d,
                stubs=True,
            )
        return d

    def validate_tree(self, tree: dict[str, str]) -> None:
        """Reject a proposal whose composed template would kill the job pre-trial."""
        with tempfile.TemporaryDirectory() as td:
            staged = HT.write_tree(OPENCODE_SURFACE, tree, Path(td) / "h")
            OC.assert_jinja_safe(OC.compose_prompt(staged))

    def preflight(self) -> None:
        """Check the runner checkout, Docker and the installed datasets before a rollout."""
        if shutil.which("uv") is None:
            raise RuntimeError(
                "`uv` is not on PATH; lingee-bench batches run as `uv run harbor run` inside "
                "its own checkout (see docs/BENCHMARKS.md)"
            )
        repo = config.lb_repo()
        if not (repo / "src" / "lg_bench" / "agents").is_dir():
            raise RuntimeError(
                f"the lingee-bench checkout at {repo} is missing src/lg_bench/agents — run "
                f"harness2-setup lb --subset {self.subset}"
            )
        if not (repo / "src" / "lg_bench" / "agents" / "harness.py").is_file():
            raise RuntimeError(
                f"{repo}/src/lg_bench/agents/harness.py is missing — run harness2-setup lb to "
                f"copy the harness agent overlay"
            )
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is required: every lingee-bench rollout builds task and sidecar "
                "containers in its own compose project"
            )
        if not self.subset_info.tasks_dir.is_dir():
            raise RuntimeError(
                f"no {self.subset} tasks under {self.subset_info.tasks_dir}; download the "
                f"dataset (harness2-setup lb --subset {self.subset})"
            )
        if not config.lb_judge_model():
            raise RuntimeError(
                "HARNESS2_JUDGE_MODEL is required: both lingee-bench suites merge a rule channel "
                "with a judge channel, and an absent judge marks the run unscored"
            )
        discover("lb", self.subset)


def spec_for(subset: str) -> BenchSpec:
    s = SUBSETS[subset]
    channel = resolve_channel_model(None, None, effort="high")
    return BenchSpec(
        name=f"lb-{subset}",
        runs_root=config.RUNS / "lb" / subset,
        install_mode="home_dir_jinja",
        prompts=lingee_prompts(subset),
        channels={"proposer": channel, "selector": channel},
        domain_word=s.domain_word,
        leak_globs=s.leak_globs,
        allow_subagents=True,
        obs_trunc=50_000,
        proposer_deliverables="contents",
        state_deliverable=True,
    )


def _task_meta(path: Path) -> dict:
    import tomllib

    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return {}


def _run_dirs(label_dir: Path) -> list[Path]:
    if not label_dir.is_dir():
        return []
    return sorted(d for d in label_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))


def _trials_in(run_dirs: list[Path], task_id: str, subset: str) -> list[Path]:
    out: list[Path] = []
    for run in run_dirs:
        for d in sorted(run.iterdir()):
            m = _GRADED_TRIAL.match(d.name)
            if d.is_dir() and m and resolve_task(m.group("task"), subset) == task_id:
                out.append(d)
    return out


def _mark_matches(trial: Path, fingerprint: str) -> bool:
    m = RM.read_mark(Path(trial))
    return bool(m) and m.get("version") == RM.VERSION and m.get("harness") == fingerprint


def _label_lock(label: str) -> threading.Lock:
    with _locks_guard:
        return _label_locks.setdefault(label, threading.Lock())


_TEXT_SUFFIXES = frozenset(
    {
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".py",
        ".sql",
        ".log",
        ".xml",
        ".html",
        ".tex",
    }
)


def _readable(path: Path) -> str:
    """Extract agent-authored text from one artifact, without third-party readers."""
    body = _extract(path)
    if len(body) > DELIVERABLE_CHAR_CAP:
        body = body[:DELIVERABLE_CHAR_CAP] + f"\n\n(truncated at {DELIVERABLE_CHAR_CAP} characters)"
    return body


def _extract(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in _TEXT_SUFFIXES:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix in (".xlsx", ".xlsm", ".docx", ".pptx"):
            return _office_text(path)
    except OSError as exc:
        return f"(unreadable: {exc})"
    size = path.stat().st_size if path.exists() else 0
    return f"({suffix or 'binary'} artifact, {size} bytes; extract with the file's own reader)"


def _office_text(path: Path) -> str:
    """Pull the text out of an OOXML package: sheet rows, paragraphs or slide text."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if path.suffix.lower() == ".docx":
            return _xml_text(zf.read("word/document.xml")) if "word/document.xml" in names else ""
        if path.suffix.lower() == ".pptx":
            parts = [n for n in sorted(names) if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
            return "\n\n".join(f"# {Path(p).stem}\n{_xml_text(zf.read(p))}" for p in parts)
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared = _shared_strings(zf.read("xl/sharedStrings.xml"))
        sheets = [n for n in sorted(names) if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)]
        out = []
        for name in sheets:
            rows = _sheet_rows(zf.read(name), shared)
            out.append(f"# {Path(name).stem}\n" + "\n".join(rows))
        return "\n\n".join(out)


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_text(blob: bytes) -> str:
    root = ElementTree.fromstring(blob)
    out: list[str] = []
    for node in root.iter():
        if _strip_ns(node.tag) == "t" and node.text:
            out.append(node.text)
        elif _strip_ns(node.tag) == "p":
            out.append("\n")
    return "".join(out).strip()


def _shared_strings(blob: bytes) -> list[str]:
    root = ElementTree.fromstring(blob)
    return ["".join(t.text or "" for t in si.iter() if _strip_ns(t.tag) == "t") for si in root]


def _sheet_rows(blob: bytes, shared: list[str]) -> list[str]:
    root = ElementTree.fromstring(blob)
    rows: list[str] = []
    for row in root.iter():
        if _strip_ns(row.tag) != "row":
            continue
        cells: list[str] = []
        for cell in row:
            if _strip_ns(cell.tag) != "c":
                continue
            kind = cell.get("t")
            value = ""
            for child in cell:
                tag = _strip_ns(child.tag)
                if tag == "v" and child.text:
                    value = child.text
                elif tag == "is":
                    value = "".join(t.text or "" for t in child.iter() if _strip_ns(t.tag) == "t")
            if kind == "s" and value.isdigit():
                idx = int(value)
                value = shared[idx] if idx < len(shared) else ""
            cells.append(value)
        if any(cells):
            rows.append("\t".join(cells))
    return rows


__all__ = [
    "LbAdapter",
    "Subset",
    "SUBSETS",
    "SUBSTRATES",
    "spec_for",
    "lingee_prompts",
    "LINGEE_BASES",
    "LINGEE_DOMAINS",
    "resolve_task",
    "graded_score",
    "job_document",
    "mcp_allow_fragment",
    "label_sanitize",
    "TRAJ_MAX_CHARS",
    "EPHEMERAL_PREFIX",
]
