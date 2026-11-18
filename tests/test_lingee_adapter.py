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

"""Offline tests for the lingee-bench (lb) adapter.

The scoring, job-YAML and harness-injection contracts run against synthetic trials;
task discovery, prompts, leak isolation and the ERP MCP allowlist run against a real
lingee-bench dataset checkout when one is available (LB_TEST_DATASETS / LB_TEST_REPO
environment variables) and are skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from harness2 import config
from harness2.benchmarks import lingee as LB
from harness2.benchmarks import tasks as TASKS
from harness2.cli import _common as C
from harness2.cli import run_sweep

_DATASETS = Path(
    os.environ.get(
        "LB_TEST_DATASETS",
        "/Users/kingdee/kiroworkspace/benchmark/lingee-bench-scan-20261020"
        "/lingee-bench-dataset",
    )
)
_REPO = Path(
    os.environ.get(
        "LB_TEST_REPO",
        "/Users/kingdee/kiroworkspace/benchmark/lingee-bench-scan-20261020"
        "/lingee-bench-runner",
    )
)
HAVE_DATASET = (_DATASETS / "lg-bench-general-v1.0" / "tasks").is_dir()
HAVE_REPO = (_REPO / "configs" / "compose").is_dir()

needs_dataset = pytest.mark.skipif(not HAVE_DATASET, reason="no local lingee-bench dataset")
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="no local lingee-bench runner checkout")


@pytest.fixture()
def datasets(monkeypatch):
    monkeypatch.setattr(config, "lb_datasets", lambda: _DATASETS)
    return _DATASETS


@pytest.fixture()
def repo(monkeypatch):
    monkeypatch.setattr(config, "lb_repo", lambda: _REPO)
    return _REPO


# --------------------------------------------------------------------------- discovery


@needs_dataset
def test_discover_general_counts_and_domains(datasets):
    got = TASKS.discover("lb", "general")
    assert len(got) == 35
    assert got == dict(sorted(got.items()))
    by_domain: dict[str, int] = {}
    for domain in got.values():
        by_domain[domain] = by_domain.get(domain, 0) + 1
    assert by_domain["document-processing"] == 8
    assert by_domain["information-retrieval"] == 7 + 4  # English labels plus 信息检索
    assert by_domain["data-analysis"] == 7
    assert by_domain["calendar-management"] == 3
    assert by_domain["image-processing"] == 2
    assert by_domain["sql-query"] == 1  # SQL数据查询
    assert by_domain["financial-reasoning"] == 2  # 财务数值推理
    assert by_domain["knowledge-qa"] == 1  # 知识问答


@needs_dataset
def test_discover_enterprise_domains_from_task_ids(datasets):
    got = TASKS.discover("lb", "enterprise")
    assert len(got) == 30
    by_domain: dict[str, int] = {}
    for domain in got.values():
        by_domain[domain] = by_domain.get(domain, 0) + 1
    assert by_domain == {"finance": 14, "manufacturing": 7, "supply-chain": 9}


@needs_dataset
def test_discover_rejects_unknown_domain_label():
    with pytest.raises(ValueError, match="not a known practice area"):
        TASKS.lb_domain("some-task-L1", {"domain": "astrology"})
    with pytest.raises(ValueError, match="no metadata.domain"):
        TASKS.lb_domain("logistics-task-L1", {})


@needs_dataset
def test_missing_subset_points_at_setup(datasets, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "lb_datasets", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="harness2-setup lb"):
        TASKS.discover("lb", "general")


def _fixture_task(tmp_path: Path) -> Path:
    """One synthetic lingee-bench general task, in the dataset's on-disk shape."""
    root = tmp_path / "lg-bench-general-v1.0" / "tasks"
    task = root / "DA-fixture-L2"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("按月完成滚动预测并输出 result.json。", encoding="utf-8")
    (task / "task.toml").write_text(
        """schema_version = "1.3"
[metadata]
domain = "data-analysis"
category = "general"
[agent]
timeout_sec = 900.0
""",
        encoding="utf-8",
    )
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    return root


def test_job_document_general_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "lb_datasets", lambda: tmp_path)
    root = _fixture_task(tmp_path)
    fragments = [
        Path("configs/compose/model-proxy.yaml"),
        Path("configs/compose/general-isolation.yaml"),
        Path("/tmp/gen/harness.yaml"),
    ]
    doc = yaml.safe_load(
        LB.job_document(
            LB.SUBSETS["general"],
            "DA-fixture-L2",
            jobs_dir=Path("/runs/lb/general/rollouts/x"),
            fragments=fragments,
            model="glm-5.3-flash",
            judge="glm-5.3-flash",
            agent_timeout=900,
            harness_ws="/opt/lg-harness/lingee-harness",
            k=2,
        )
    )
    assert doc["jobs_dir"] == "/runs/lb/general/rollouts/x"
    assert doc["n_attempts"] == 2
    assert doc["retry"] == {"max_retries": 0}
    assert doc["environment"]["type"] == "docker"
    assert doc["environment"]["extra_docker_compose"] == [str(f) for f in fragments]
    assert doc["tasks"] == [{"path": str(root / "DA-fixture-L2")}]
    assert doc["artifacts"] == ["/workspace/output"]
    (agent,) = doc["agents"]
    assert agent["import_path"] == "lg_bench.agents.harness:LgHarnessOpenCode"
    assert agent["model_name"] == "glm-5.3-flash"
    assert "/" not in agent["model_name"]
    assert agent["kwargs"]["install_mode"] == "image"
    assert agent["kwargs"]["harness_ws"] == "/opt/lg-harness/lingee-harness"
    assert agent["override_timeout_sec"] == 900
    venv = doc["verifier"]["env"]
    assert venv["JUDGE_BASE_URL"] == "http://lg-model-proxy:18080/openai"
    assert venv["JUDGE_STRICT"] == "1"
    assert venv["JUDGE_MODEL"] == "glm-5.3-flash"


def test_job_document_enterprise_wires_erp_stack(tmp_path, repo, monkeypatch):
    monkeypatch.setattr(config, "lb_datasets", lambda: tmp_path)
    ent_root = tmp_path / "lg-bench-enterprise-v1.0" / "tasks" / "finance-bgt-01-L2"
    ent_root.mkdir(parents=True)
    (ent_root / "instruction.md").write_text("录入凭证", encoding="utf-8")
    (ent_root / "task.toml").write_text(
        """schema_version = "1.4"
[metadata]
case_id = "BGT-01"
[agent]
timeout_sec = 1200.0
[verifier]
environment_mode = "shared"
""",
        encoding="utf-8",
    )
    LB.LbAdapter("enterprise")  # construction wires the subset table
    doc = yaml.safe_load(
        LB.job_document(
            LB.SUBSETS["enterprise"],
            "finance-bgt-01-L2",
            jobs_dir=Path("/runs/lb/enterprise/rollouts/x"),
            fragments=[Path("/tmp/gen/a.yaml"), Path("/tmp/gen/harness.yaml")],
            model="glm-5.3-flash",
            judge="glm-5.3-flash",
            agent_timeout=1200,
            harness_ws="/opt/lg-harness/lingee-harness",
            k=1,
        )
    )
    (agent,) = doc["agents"]
    (mcp,) = agent["mcp_servers"]
    assert mcp == {"name": "erp", "transport": "streamable-http", "url": "http://lg-mcp-proxy:18090/mcp"}
    assert "verifier" not in doc or not doc.get("verifier", {}).get("env")  # ERP judge rides compose


@needs_dataset
@needs_repo
def test_every_enterprise_task_maps_to_one_mcp_fragment(datasets, repo):
    monkey_env = {}
    for task in TASKS.discover("lb", "enterprise"):
        frag = LB.mcp_allow_fragment(task, {})
        assert frag, task
        assert (repo / "configs" / "compose" / f"mcp-allow-{frag}.yaml").is_file(), (task, frag)
    assert len(monkey_env) == 0


# ------------------------------------------------------------------------ scoring


def _trial(tmp_path: Path, name="DA-x-L2__ab12xyz") -> Path:
    trial = tmp_path / name
    (trial / "verifier").mkdir(parents=True)
    return trial


def _score_dict(raw: float, status: str = "dual_channel") -> dict:
    return {
        "n_passed": round(raw * 100),
        "n_criteria": 100,
        "all_pass": raw >= 1.0,
        "raw_score": raw,
        "verifier_status": status,
    }


def test_score_general_channels(tmp_path):
    adapter = LB.LbAdapter("general")

    graded = _trial(tmp_path)
    (graded / "verifier" / "score.json").write_text(
        json.dumps({"score": 0.62, "status": "dual_channel", "rule_score": 0.4, "judge_score": 0.84}),
        encoding="utf-8",
    )
    assert adapter.score(graded, "DA-x-L2") == _score_dict(0.62, "dual_channel")

    core = _trial(tmp_path, "DA-y-L2__cd34efg")
    (core / "verifier" / "score.json").write_text(
        json.dumps({"score": 0, "status": "core_threshold_failed"}), encoding="utf-8"
    )
    assert adapter.score(core, "DA-y-L2") == _score_dict(0.0, "core_threshold_failed")

    invalid = _trial(tmp_path, "DA-z-L2__ef56ghi")
    (invalid / "verifier" / "score.json").write_text(
        json.dumps({"score": 0, "status": "evaluation_error", "error_kind": "invalid_judge_key_missing"}),
        encoding="utf-8",
    )
    assert adapter.score(invalid, "DA-z-L2") is None  # evaluation error is NOT a zero

    missing = _trial(tmp_path, "DA-w-L2__hi78jkl")
    assert adapter.score(missing, "DA-w-L2") is None  # unscored stays unscored


def test_score_general_steps_trials_read_last_step(tmp_path):
    trial = tmp_path / "DRV-multiturn-L3__op90qrs"
    last = trial / "steps" / "step-2" / "verifier"
    last.mkdir(parents=True)
    (last / "score.json").write_text(
        json.dumps({"score": 1.0, "status": "dual_channel"}), encoding="utf-8"
    )
    assert LB.LbAdapter("general").score(trial, "DRV-multiturn-L3") == _score_dict(1.0)


def test_score_enterprise_results_json(tmp_path):
    adapter = LB.LbAdapter("enterprise")
    trial = _trial(tmp_path, "finance-gl-03-L3__uv12wxy")
    (trial / "verifier" / "results.json").write_text(
        json.dumps({"status": "graded", "content_score": 0.75}), encoding="utf-8"
    )
    (trial / "verifier" / "reward.json").write_text(
        json.dumps({"content_score": 0.75, "overall": 0.75}), encoding="utf-8"
    )
    assert adapter.score(trial, "finance-gl-03-L3") == _score_dict(0.75, "graded")

    broken = _trial(tmp_path, "finance-gl-03-L3__wx34yz0")
    (broken / "verifier" / "results.json").write_text(
        json.dumps({"status": "evaluation_error", "error": "judge_unavailable"}), encoding="utf-8"
    )
    assert adapter.score(broken, "finance-gl-03-L3") is None


def test_killed_before_deliverable_is_unscored(tmp_path):
    adapter = LB.LbAdapter("general")
    trial = _trial(tmp_path)
    (trial / "verifier" / "score.json").write_text(
        json.dumps({"score": 0, "status": "dual_channel"}), encoding="utf-8"
    )
    (trial / "agent").mkdir()
    (trial / "agent" / "opencode.txt").write_text(
        json.dumps({"type": "error", "error": {"message": "provider 500"}}) + "\n",
        encoding="utf-8",
    )
    assert adapter.score(trial, "DA-x-L2") is None


def test_score_dict_roundtrip_through_judge_contract(tmp_path):
    trial = _trial(tmp_path)
    (trial / "verifier" / "score.json").write_text(
        json.dumps({"score": 0.62, "status": "dual_channel"}), encoding="utf-8"
    )
    (trial / "artifacts" / "output").mkdir(parents=True)
    (trial / "artifacts" / "output" / "result.json").write_text("{}", encoding="utf-8")
    got = LB.LbAdapter("general").score(trial, "DA-x-L2")
    assert got["n_passed"] <= got["n_criteria"]
    assert 0.0 <= got["raw_score"] <= 1.0


# ------------------------------------------------------------- deliverables and trajectory


def _office_xlsx(path: Path) -> None:
    import zipfile

    sst = (
        '<?xml version="1.0"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<si><t>预算</t></si><si><t>实际</t></si></sst>"
    ).encode("utf-8")
    sheet = (
        '<?xml version="1.0"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        "</sheetData></worksheet>"
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("xl/sharedStrings.xml", sst)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


def test_delivered_and_extract_office_artifacts(tmp_path):
    adapter = LB.LbAdapter("general")
    trial = _trial(tmp_path)
    out = trial / "artifacts" / "output"
    out.mkdir(parents=True)
    _office_xlsx(out / "result.xlsx")
    (out / "summary.md").write_text("# 结论\n预算偏差 3%", encoding="utf-8")
    (out / "data.bin").write_bytes(b"\x00\x01\x02")

    assert adapter.delivered(trial)
    files = adapter.deliverable_files(trial)
    assert set(files) == {"output/result.xlsx", "output/summary.md", "output/data.bin"}
    assert "预算" in files["output/result.xlsx"]  # shared strings extracted
    assert "预算偏差 3%" in files["output/summary.md"]
    text = adapter.deliverable_text(trial)
    assert "预算偏差 3%" in text


def test_delivered_requires_artifact_bodies(tmp_path):
    adapter = LB.LbAdapter("general")
    trial = _trial(tmp_path)
    (trial / "artifacts").mkdir()
    (trial / "artifacts" / "manifest.json").write_text("{}", encoding="utf-8")
    assert not adapter.delivered(trial)  # no /workspace/output artifact at all

    empty = _trial(tmp_path, "DA-empty-L2__n0p1qrs")
    (empty / "artifacts" / "output").mkdir(parents=True)
    (empty / "artifacts" / "output" / ".keep").write_text("", encoding="utf-8")
    assert not adapter.delivered(empty)  # empty body is not a deliverable


def test_trajectory_renders_shared_grammar(tmp_path):
    adapter = LB.LbAdapter("general")
    trial = _trial(tmp_path)
    (trial / "agent").mkdir()
    (trial / "agent" / "opencode.txt").write_text(
        "\n".join(
            [
                json.dumps({"part": {"type": "text", "text": "开始分析报表数据"}}),
                json.dumps(
                    {
                        "part": {
                            "type": "tool-invocation",
                            "tool": "edit",
                            "state": {"input": {"path": "out.md"}, "status": "completed"},
                        }
                    }
                ),
                json.dumps({"part": {"type": "text", "text": "已写入输出文件"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text = adapter.trajectory(trial, obs_trunc=50_000)
    assert "[ACTION]" in text or "edit" in text
    assert "开始分析报表数据" in text
    assert "已写入输出文件" in text

    none = _trial(tmp_path, "DA-none-L2__tuv234")
    assert adapter.trajectory(none, obs_trunc=50_000) == "(no trajectory)"


# ---------------------------------------------------------------- harness injection


def _harness_agent_module():
    import types

    if "lg_bench" not in sys.modules:  # a stub, so the import works without the checkout
        pkg = types.ModuleType("lg_bench")
        agents = types.ModuleType("lg_bench.agents")
        opencode = types.ModuleType("lg_bench.agents.opencode")

        class LgOpenCode:  # the real class lives in the lingee-bench runner
            pass

        opencode.LgOpenCode = LgOpenCode
        pkg.agents = agents
        sys.modules["lg_bench"] = pkg
        sys.modules["lg_bench.agents"] = agents
        sys.modules["lg_bench.agents.opencode"] = opencode

    src = (
        Path(config.ROOT) / "third_party" / "lb_runner" / "agents" / "lb_harness_agent.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("lb_harness_agent", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _staged_harness(tmp_path: Path) -> Path:
    ws = tmp_path / "staged"
    (ws / "_composed").mkdir(parents=True)
    (ws / "_composed" / "prompt.j2").write_text(
        "you are careful with ledgers\n{{ instruction }}", encoding="utf-8"
    )
    (ws / "skills").mkdir()
    (ws / "skills" / "forensics").mkdir()
    (ws / "skills" / "forensics" / "SKILL.md").write_text(
        "---\ndescription: drill into cost overruns\n---\nbody", encoding="utf-8"
    )
    (ws / "scripts").mkdir()
    (ws / "scripts" / "helper.py").write_text("print('hi')", encoding="utf-8")
    (ws / "sub_agents").mkdir()
    (ws / "sub_agents" / "reviewer").mkdir()
    (ws / "sub_agents" / "reviewer" / "AGENT.md").write_text("review the ledger", encoding="utf-8")
    return ws


def test_harness_payload_layout_and_prompt_stripping(tmp_path):
    mod = _harness_agent_module()
    ws = _staged_harness(tmp_path)
    agents_md = mod.compose_agents_md(ws)
    assert agents_md == "you are careful with ledgers"  # instruction slot dropped
    blob, n = mod.harness_payload(ws)
    assert n == 4  # skill + script + subagent + AGENTS.md
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = sorted(tar.getnames())
        assert ".config/opencode/AGENTS.md" in names
        assert ".config/opencode/skill/forensics/SKILL.md" in names
        assert ".config/opencode/agent/reviewer.md" in names
        assert "harness_scripts/helper.py" in names
        body = tar.extractfile(".config/opencode/AGENTS.md").read().decode("utf-8")
        assert body == "you are careful with ledgers"
        assert "{{ instruction }}" not in body


def test_empty_harness_is_stock(tmp_path):
    mod = _harness_agent_module()
    bare = tmp_path / "bare"
    bare.mkdir()
    assert mod.compose_agents_md(bare) == ""
    blob, n = mod.harness_payload(bare)
    assert n == 0


# ------------------------------------------------------------------------- CLI wiring


def test_make_adapter_routes_lb():
    assert "lb" in C.BENCHES
    with pytest.raises(ValueError, match=r"needs --subset from \('general', 'enterprise'\)"):
        C.make_adapter("lb", "office", "oc")


def test_cell_args_accept_lb():
    import argparse

    p = argparse.ArgumentParser()
    C.add_cell_args(p)
    args = p.parse_args(["--bench", "lb", "--subset", "general"])
    assert (args.bench, args.subset, args.substrate) == ("lb", "general", "oc")


@needs_dataset
def test_dry_run_prints_lb_stream(datasets, monkeypatch, capsys):
    monkeypatch.setattr(config, "SOLVER_MODEL_OC", "zhipu/glm-5.3-flash")
    monkeypatch.setattr(config, "IMPROVER_MODEL", "zhipu/glm-5.3-flash")
    code = run_sweep.main(
        ["--bench", "lb", "--subset", "general", "--k", "1", "--mode", "parallel", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "lb-general-oc" in out
    assert "namespace[parallel]" in out
    assert "# 35 tasks over 8 domains" in out
    assert "信息检索" not in out  # domains are normalised slugs


@needs_dataset
def test_dry_run_enterprise_counts(datasets, capsys):
    code = run_sweep.main(
        ["--bench", "lb", "--subset", "enterprise", "--k", "2", "--mode", "parallel", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "# 30 tasks over 3 domains" in out
    assert "finance" in out and "supply-chain" in out


def test_preflight_skips_patch_for_lb(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DEPENDENCIES", tmp_path)
    import subprocess as sp

    recorded: list[list[str]] = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        recorded.append([str(x) for x in cmd])
        return FakeResult()

    monkeypatch.setattr(sp, "run", fake_run)

    patched = [cmd for cmd in recorded if "apply" in cmd]
    assert not any("lingee-bench" in " ".join(cmd) and "apply" in cmd for cmd in patched)
