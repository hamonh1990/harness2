# Benchmark guides

Complete the [installation and credentials](SETUP.md) first. Each
benchmark installs separately; install the ones you intend to run. Every command
below takes `--substrate cx` to use the
[Codex executor](SETUP.md#codex-executor) instead of OpenCode. Choose the
recursion mode with `--mode parallel` or `--mode sequential` and the task-level
recursion budget with `--k`. Keep both identical across the run, judge, and
report commands.

## LAB

Upstream: [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs).
Legal work tasks with source documents and rubric-based grading. The adapter
groups tasks by practice area and removes criteria from the material passed to
the improver. Needs Pandoc and Poppler.

```bash
harness2-setup lab
harness2-check --bench lab
harness2 --bench lab --k 1 --mode parallel
harness2-judge --bench lab --k 1 --mode parallel
harness2-report --bench lab --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

Tasks arrive with the pinned checkout at
`dependencies/harvey-labs/tasks/<practice-area>/<task>/task.json`, and the adapter
discovers all installed tasks; no separate task list is required. Setup installs
the LAB dependency environment, the runner overlays, and the included Vertex
judge patch.

The judge reads outputs only after selection. Reports use Partial (fraction of
criteria passed) and All-pass (all criteria passed). A failed judge remains
unscored and writes its output to `judge.log` in the run directory; a Vertex
`429 RESOURCE_EXHAUSTED` there means quota, so rerun `harness2-judge` later.
Changing the judge model changes the measurement. Outputs are under
`runs/lab/`.

## JobBench

Upstream: [Job-Bench/job-bench-eval](https://github.com/Job-Bench/job-bench-eval).
Data: [JobBench/job-bench](https://huggingface.co/datasets/JobBench/job-bench).
Professional tasks, with the library grouped by the dataset's occupation
directories.

```bash
harness2-setup jb
harness2-check --bench jb
harness2 --bench jb --k 1 --mode parallel
harness2-judge --bench jb --k 1 --mode parallel
harness2-report --bench jb --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

Setup downloads only the main dataset from Hugging Face — public and ungated, so
no account, token, or terms acceptance is required — records its revision, and
places tasks at `dependencies/job-bench-eval/dataset/main/`. It also installs the
document extraction dependencies and a judge patch that handles reasoning effort
and empty completions. Rubrics stay outside the improver's evidence.

With a Gemini `HARNESS2_JUDGE_MODEL`, the judge uses the project's Vertex
OpenAI-compatible endpoint and Application Default Credentials. For a separate
judge provider, set `HARNESS2_JUDGE_BASE_URL`, `HARNESS2_JUDGE_API_KEY`, and the
model ID that endpoint accepts. The reported score uses the upstream rubric
weights; unscored tasks are reported separately. Results are under `runs/jb/`.

Note: JobBench data, including `RUBRICS.json`, is public on Hugging Face, so an
executor with web access may retrieve it. Block those hosts during runs as
described under [Web access](EXPERIMENTS.md#web-access).

## WorkBuddy-Bench

Upstream: [Tencent/workbuddy-bench](https://github.com/Tencent/workbuddy-bench).
Data: [tencent/workbuddy-bench](https://huggingface.co/datasets/tencent/workbuddy-bench).
Office, web, and code subsets. Needs Docker.

The upstream custom Tencent license includes a territorial limitation; read
[the license](https://github.com/Tencent/workbuddy-bench/blob/main/LICENSE)
before downloading or using the benchmark. Derived runner files and patches
retain that license.

```bash
docker info
harness2-setup wb --subset office
harness2-check --bench wb --subset office
harness2 --bench wb --subset office --k 1 --mode parallel
harness2-judge --bench wb --subset office --k 1 --mode parallel
harness2-report --bench wb --subset office --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

`harness2-setup wb` without `--subset` installs all three subsets in one run;
pass `--subset` to install only the one you intend to run. Every other command
takes one subset at a time.

Setup installs the pinned Harbor-based runtime, applies the included adapter,
mount, and credential/transport patches, copies the harness runners, creates
solver and judge routes from `.env`, downloads the requested subset with the
upstream fetch script, and builds the OpenCode harness image. The public Vertex
setup uses a `google-vertex/gemini...` executor and a bare `gemini...` judge ID;
no model YAML or account-specific proxy configuration needs to be copied from
another checkout. Setup refuses to run unless `HARNESS2_SOLVER_MODEL` is a
`google-vertex/gemini...` ID and `HARNESS2_JUDGE_MODEL` a bare `gemini...` one,
with `GOOGLE_CLOUD_PROJECT` set; `--dry-run` reports the requirement and still
prints the rest of the plan. The benchmark's job-local proxy refreshes ADC tokens
and preserves Gemini tool-call thought signatures.

Setup patches the upstream fetch script to download with `curl` (then `wget`,
then `hf`) and to require the dataset repository's `SHA256SUMS`: an archive that
cannot be verified is not unpacked. A verified download prints
`wb-bench-<subset>-v1.0.tar.gz: OK`.

Tasks are discovered from `task.toml` and `instruction.md` under
`dependencies/workbuddy-bench/datasets/wb-bench-<subset>-v1.0/tasks/`. Libraries use
`metadata.role` on code and `metadata.category` on office and web. Each rollout
uses a fresh benchmark container. The adapter reads the verifier's `score.json`;
the improver never receives verifier output. Missing verifier results remain
unscored. Runs live under `runs/wb/<subset>/`.

For Codex, start the bridge with `harness2-bridge --host 0.0.0.0` so that the containers can reach it.

## lingee-bench (lb)

Upstream: [kingdee/lingee-bench](https://github.com/kingdee/lingee-bench).
Data: [KD-Lingee/lingee-bench](https://huggingface.co/datasets/KD-Lingee/lingee-bench).
General and enterprise subsets. Needs Docker.

```bash
docker info
harness2-setup lb --subset general
harness2-check --bench lb --subset general
harness2 --bench lb --subset general --k 1 --mode parallel
harness2-judge --bench lb --subset general --k 1 --mode parallel
harness2-report --bench lb --subset general --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

`harness2-setup lb` without `--subset` installs both subsets in one run; pass
`--subset` to install only the one you intend to run. Every other command takes one
subset at a time. Only the OpenCode substrate is supported: the lingee-bench runner
ships OpenCode, Claude Code, Pi and DeepSeek harnesses, and only the OpenCode one takes
an editable harness tree.

Setup clones the pinned runner, copies the harness-adapter overlay into
`src/lg_bench/agents/harness.py`, syncs its lockfile, downloads the requested
subsets from Hugging Face (recording the resolved revision), verifies them with the
runner's own `scripts/check_release.py`, and builds the model proxy and OpenCode
harness images. The enterprise subset additionally builds the offline ERP mock and
MCP proxy images, whose instance data comes from the download. The model routes come
from `.env`: `HARNESS2_SOLVER_MODEL` keeps OpenCode's `provider/model` syntax for the
proposer/selector channels, while the executor's model proxy admits the bare model
name; set `HARNESS2_LB_ENV_FILE` to the runner's own env file (LG_OPENAI_*,
LG_ANTHROPIC_*) for the proxy sidecar credentials.

Tasks are discovered from `task.toml` and `instruction.md` under
`dependencies/lingee-bench/datasets/lg-bench-<subset>-v1.0/tasks/`. Libraries use
`metadata.domain` on the general subset (normalized to one slug per practice area;
the enterprise subset derives it from the task-id prefix: finance, manufacturing,
supply-chain). Each rollout is one Harbor job with a fresh benchmark container. The
adapter reads the verifier's own `score.json` (general) or `results.json`
(enterprise): `dual_channel` and `core_threshold_failed` are graded (the latter is a
genuine zero for a missing core artifact), anything else — including a failed or
unconfigured judge — stays unscored rather than becoming a zero, and the improver
never receives verifier output. Runs live under `runs/lb/<subset>/`.

The harness tree is bind-mounted into the container and installed by the overlay
agent as OpenCode's global `AGENTS.md` plus its skills, scripts, plugins and
subagents; an empty tree reproduces the stock agent. The dataset is public, so an
executor with web access may retrieve the task packages; the general tasks declare
`verifier.network_mode = "public"`, so block the dataset hosts during runs as
described under [Web access](EXPERIMENTS.md#web-access) if that matters for your
setup.
