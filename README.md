# ai-coding-agent — a small, readable coding agent

A command-line **coding agent** that solves a task by looping through
**think → act → observe** steps: it asks a model for the next action, runs a
tool (read/write files, search, run shell commands) inside a **sandboxed
workspace**, feeds the result back, and repeats until it's done.

It's built to be *read*: the agent loop is an explicit **LangGraph** state
machine, tools are ~100 lines, and the model layer is a tiny text-in/text-out
interface. It runs **offline with zero API keys** out of the box (a deterministic
mock model drives the loop), and switches to **OpenAI, Anthropic, or a local
Ollama model** with one environment variable.

```console
$ python -m agent "Create greet.py with a greet(name) function, then run it"
Coding agent (model: mock, workspace: ./workspace)
Task: Create greet.py with a greet(name) function, then run it

[1] think Create greet.py with the requested function and a runnable __main__.
    → write_file(path='greet.py', content='"""Tiny module produced by the …')
    wrote 178 chars to greet.py
[2] think Run greet.py to confirm it executes cleanly.
    → run_shell(command='python greet.py')
    exit=0
    Hello, World!
✓ Done Created `greet.py` and verified it runs (prints a greeting).
```

## Architecture

The agent is a **LangGraph** graph. Splitting *think* (decide) from *act*
(execute) keeps the control flow explicit — the graph literally *is* the policy.

```
                    ┌──────────── model reply (JSON) ────────────┐
                    │                                             │
   START ─▶ think ──┴─▶ route ──"tool"──▶ act ──observation──▶ think ...
              ▲                │
              │             "final" / budget spent
              │                │
              └──────────      ▼
                          finalize ─▶ END
```

- **think** builds the prompt (system + task + running transcript), calls the
  model, and parses its reply into an action.
- **route** sends tool calls to `act`, and a `final` answer (or a spent step
  budget) to `finalize`.
- **act** executes the tool in the sandbox and appends the observation to the
  transcript, which the next `think` sees.

**Provider-neutral protocol.** Instead of vendor-specific tool-calling APIs, the
model replies with a single JSON object:

```json
{"thought": "…", "action": "tool", "tool": "write_file", "args": {"path": "greet.py", "content": "…"}}
{"thought": "…", "action": "final", "answer": "…"}
```

Parsing is lenient (tolerates code fences / prose), so the *same* prompt works
across OpenAI, Anthropic, Ollama, and the offline mock.

## Repository layout

```
agent/
├── graph.py        # LangGraph think/act/finalize loop  ← the core
├── tools.py        # sandboxed toolbox (read/write/list/grep/run_shell)
├── prompts.py      # system prompt + JSON protocol contract
├── state.py        # AgentState (TypedDict)
├── cli.py          # `python -m agent "task"`, streams the loop live
└── llm/            # tiny text-in/text-out model layer
    ├── base.py         # Message, ChatModel
    ├── providers.py    # OpenAI / Anthropic / Ollama (httpx, no SDKs)
    ├── mock.py         # deterministic offline policy (zero keys)
    └── factory.py      # get_chat_model() from LLM_PROVIDER
tests/              # hermetic pytest suite (uses the mock model)
```

## Quick start (local, no keys)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m agent                       # runs the built-in demo task
python -m agent "List the Python files here and summarize them" --workdir .
```

The default `--workdir` is `./workspace` (created on demand). Files the agent
writes and commands it runs are confined to that directory.

## Local setup (all options)

No database and no Docker are required — the only choice is which model drives
the loop, selected by environment variables (all listed in
[`.env.example`](.env.example), loaded automatically from a `.env`). No keys are
ever committed.

```bash
# a) Offline (default): deterministic mock policy, zero setup
export LLM_PROVIDER=mock
python -m agent

# b) Ollama — local & free. Install from https://ollama.com, then:
ollama pull llama3.1
export LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1
python -m agent "Write a factorial function in math_utils.py and run it"
#   ...or run Ollama in Docker instead of installing it:
#   docker run -d --name ollama -p 11434:11434 ollama/ollama
#   docker exec -it ollama ollama pull llama3.1
#   export LLM_PROVIDER=ollama OLLAMA_HOST=http://localhost:11434

# c) Hosted API (needs a key)
export LLM_PROVIDER=openai    OPENAI_API_KEY=sk-...          # or:
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
python -m agent "Refactor utils.py to add type hints and a docstring" --workdir ./workspace
```

## Safety model

The agent can write files and run shell commands, so it's deliberately fenced in:

- **Path jail.** Every path is resolved and must live inside the workspace root;
  `..` traversal and absolute paths are rejected (`tools.Toolbox._safe`).
- **Bounded shell.** `run_shell` runs with the workspace as its cwd, a 30s
  timeout, and truncated output.
- **Step budget.** The loop stops after `--max-iterations` (default 12) even if
  the model never says it's finished.

This is enough to safely demo and develop against a scratch directory. It is
**not** a security boundary for untrusted input — for that you'd run the tools in
a container/VM with no network and a read-only base image (noted as future work).

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is **hermetic**: it drives the real graph and tools with the mock
model, so there are no network calls and results are deterministic. It covers
sandbox path-safety, each tool, protocol parsing, and two end-to-end runs.

## Design decisions & tradeoffs

- **LangGraph, with think/act split.** A visible state machine beats a hidden
  `while` loop for reviewability, and makes adding a step (e.g. a `reflect` node
  or human-in-the-loop approval before `act`) a one-line change.
- **JSON protocol over native tool-calling.** One prompt for every provider and
  trivial offline testing, at the cost of not using each vendor's structured
  tool API. For a single-provider production agent, native tool-calling +
  schema validation would be the better call.
- **Deterministic mock as the default model.** The project runs and its tests
  pass with zero setup or keys. The mock is a scripted policy, not a real coder —
  point `LLM_PROVIDER` at a real model for genuine work.
- **httpx instead of vendor SDKs.** Small, dependency-light installs and nothing
  to keep in lockstep with fast-moving client libraries.

## Notes

Built as a focused iteration to show agent design (a tool-use loop, sandboxing,
provider abstraction) rather than breadth. Where it stops short — a container
sandbox, native tool-calling, multi-file diffs/patches, retrieval over the repo —
those are deliberate scoping choices.
