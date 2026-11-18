# SPDX-License-Identifier: Apache-2.0
"""LgOpenCode subclass that installs the on-demand harness components from a bind-mounted
harness workspace; an empty workspace reproduces the stock lingee-bench agent.

The job YAML mounts the harness tree at `/opt/lg-harness/lingee-harness` (see
`harness2/benchmarks/lingee.py`) and passes its container path as the `harness_ws`
agent kwarg. This agent installs the components OpenCode itself loads:

- `AGENTS.md` from the composed prompt (minus Harbor's `{{ instruction }}` slot, which
  the task instruction already fills) into `~/.config/opencode/AGENTS.md`;
- `skills/`, `scripts/`, `plugin/` and `sub_agents/` into their OpenCode config dirs,
  following the same layout the WorkBuddy harness agent installs.

Everything arrives as ONE base64 tarball executed through `exec_as_agent`, so there is
no per-file transport and no host path leaks into the container.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

from lg_bench.agents.opencode import LgOpenCode

_CONFIG_DIR = ".config/opencode"
_SKILL_SUBDIR = f"{_CONFIG_DIR}/skill"
_AGENT_SUBDIR = f"{_CONFIG_DIR}/agent"
_PLUGIN_SUBDIR = f"{_CONFIG_DIR}/plugin"
_SCRIPT_SUBDIR = "harness_scripts"
_PROMPT_REL = "_composed/prompt.j2"
INSTRUCTION_PLACEHOLDER = "{{ instruction }}"


def compose_agents_md(ws: Path) -> str:
    """The always-on harness text, with Harbor's instruction slot dropped: the run
    prompt already carries the task instruction, so keeping the slot would repeat it."""
    p = Path(ws) / _PROMPT_REL
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace").replace(
        INSTRUCTION_PLACEHOLDER, ""
    ).strip()


def harness_payload(ws: Path) -> tuple[bytes, int]:
    """The harness tree as one gzipped tar targeting `$HOME/<subdir>`; (bytes, nfiles).

    Subagents flatten (sub_agents/<name>/AGENT.md -> agent/<name>.md) the way OpenCode
    wants them; the composed prompt rides along as the global AGENTS.md.
    """
    ws = Path(ws)
    payload = io.BytesIO()
    n = 0
    agents_md = compose_agents_md(ws)
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        for src_name, dest, flatten in (
            ("skills", _SKILL_SUBDIR, False),
            ("scripts", _SCRIPT_SUBDIR, False),
            ("plugin", _PLUGIN_SUBDIR, False),
            ("sub_agents", _AGENT_SUBDIR, True),
        ):
            src = ws / src_name
            if not src.is_dir():
                continue
            if flatten:
                for p in sorted(src.glob("*/AGENT.md")):
                    tar.add(p, arcname=f"{dest}/{p.parent.name}.md")
                    n += 1
            else:
                for p in sorted(src.rglob("*")):
                    if p.is_file():
                        tar.add(p, arcname=f"{dest}/{p.relative_to(src)}")
                        n += 1
        if agents_md:
            blob = agents_md.encode("utf-8")
            info = tarfile.TarInfo(f"{_CONFIG_DIR}/AGENTS.md")
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
            n += 1
    return payload.getvalue(), n


class LgHarnessOpenCode(LgOpenCode):
    """OpenCode under the lingee-bench proxy, plus a mounted harness workspace."""

    def __init__(self, logs_dir: Path, *args, **kwargs):
        ws = kwargs.pop("harness_ws", None)
        self._harness_ws: Path | None = Path(ws) if ws else None
        super().__init__(logs_dir, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "opencode-harness"

    def _inject_plugins(self, config_b64: str) -> str:
        """Add the harness's plugins to the generated opencode.json (specs resolve
        against the config file's own directory); any preset list is preserved."""
        ws = self._harness_ws
        root = ws / "plugin" if ws else None
        names = sorted(p.name for p in root.glob("*.ts")) if root and root.is_dir() else []
        if not names:
            return config_b64
        cfg = json.loads(base64.b64decode(config_b64).decode())
        existing = list(cfg.get("plugin") or [])
        for plugin in names:
            spec = f"./plugin/{plugin}"
            if spec not in existing:
                existing.append(spec)
        cfg["plugin"] = existing
        return base64.b64encode(json.dumps(cfg).encode()).decode()

    async def install(self, environment) -> None:
        await super().install(environment)
        if self._harness_ws is not None:
            await self._materialize_harness(environment)

    async def _materialize_harness(self, environment) -> None:
        """Install every on-demand component into the agent's home, as one base64 tar."""
        ws = self._harness_ws
        if ws is None or not ws.is_dir():
            return
        blob, n = harness_payload(ws)
        if n == 0:
            return
        encoded = base64.b64encode(blob).decode()
        home_fix = 'export HOME="$(getent passwd "$(id -u)" | cut -d: -6)"; '
        dirs = " ".join(
            f'"$HOME/{d}"' for d in (_SKILL_SUBDIR, _AGENT_SUBDIR, _PLUGIN_SUBDIR, _SCRIPT_SUBDIR)
        )
        cmd = (
            home_fix
            + f"mkdir -p {dirs} && "
            + f"echo {encoded} | base64 -d | tar -xzf - -C \"$HOME\" && "
            + f'chmod -R a+rX "$HOME/{_CONFIG_DIR}" "$HOME/{_SCRIPT_SUBDIR}"'
        )
        await self.exec_as_agent(environment, command=cmd)
        log = getattr(self, "logger", None)
        if log is not None:
            log.info("materialized %d harness file(s) from %s", n, ws)
