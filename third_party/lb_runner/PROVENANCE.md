# Provenance

`third_party/lb_runner/` holds harness2-side integration files for the
[kingdee/lingee-bench](https://github.com/kingdee/lingee-bench) benchmark
(Apache-2.0). Nothing here is upstream material: the single file
`agents/lb_harness_agent.py` is harness2-authored code that `harness2-setup lb`
copies into the lingee-bench checkout as `src/lg_bench/agents/harness.py`.

It subclasses the runner's own `lg_bench.agents.opencode.LgOpenCode` agent to
install the mounted harness tree (composed prompt, skills, scripts, plugins,
subagents) into the agent container, following the same layout the WorkBuddy
harness agent installs.
