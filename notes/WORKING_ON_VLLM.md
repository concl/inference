# Working on vLLM internals

vLLM is a **uv dependency**, not a copy vendored into this repo. You work on it in a
checkout of our fork `https://github.com/concl/vllm.git`, and the result of that
work is a **branch** — which is also the pull request.

## Two things, deliberately separate

| | Where | Purpose |
|---|---|---|
| The change | a branch on `concl/vllm` | the PR |
| Which vLLM this repo installs | one line in `pyproject.toml` | reproducibility |

## Setup (once)

```bash
git clone https://github.com/concl/vllm.git ~/vllm    # the fork
git -C ~/vllm switch -c acmai/<change>                # a branch is a change
```

Install it **editable**, so edits take effect without reinstalling:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e ~/vllm
```

`VLLM_USE_PRECOMPILED=1` reuses vLLM's prebuilt C++/CUDA binaries, so only the
Python side is editable and it takes minutes rather than a full CUDA build. Drop it
when you are changing kernels and need them recompiled.

## Day to day

```bash
# ...edit ~/vllm/vllm/..., run the server, run tests...
git -C ~/vllm commit -s -am "..."      # -s = DCO sign-off, required upstream
```

Serve and benchmark from this repo, as usual:

```bash
inference serve
vllm bench serve --save-result --result-dir experiments/results --label <name>
```

## Submitting

```bash
git -C ~/vllm push -u origin acmai/<change>
```

Then open the PR against `vllm-project/vllm` on GitHub. **The PR diff is the
branch** — there is no patch file to keep in sync.

## Pinning what this repo installs

By default `uv sync` gives the PyPI release. To install a specific fork branch
instead, uncomment the source in `pyproject.toml`:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/concl/vllm.git", branch = "acmai/<change>" }
```

Two consequences:

- uv fetches from the **remote**, so the branch must be pushed before this resolves.
- A git source is **not editable**, so it needs `uv sync --reinstall-package vllm`
  after each push.

That is why the edit loop above uses the `-e ~/vllm` install, and why the
`uv.sources` line stays commented unless you specifically want the pin.

## Rules

1. **Never commit an ACM AI-specific file in the fork.** `git -C ~/vllm status` should only
   ever show your intended vLLM change. A `serve.py` or a notes file in there is a
   rebase conflict on every upstream pull, and a hunk you must strip from the PR.
2. **If it ships in the PR, it lives in the fork.** vLLM's CI runs `pytest` inside
   the checkout, so a test written in this repo cannot be submitted.
   - change: `vllm/...`
   - correctness test: `tests/kernels/...`
   - shipped microbenchmark: `benchmarks/kernels/`
3. **Sign off every commit** (`commit -s`). vLLM requires the DCO.
4. **Disclose AI assistance** in the PR description, with a `Co-authored-by:` trailer
   where applicable — vLLM's contributing guide requires it, and asks that a human
   reviews every changed line.
5. **Rebase before submitting.** vLLM's CI refuses a branch that is behind its target.

## Current branch

`acmai/mamba-drop-eagle-block` — makes `MambaManager.find_longest_cache_hit` honour
`drop_eagle_block`, so an MTP/EAGLE prefix-cache hit cannot resume from recurrent
state taken over rejected draft tokens. Committed in `~/vllm`, **not yet pushed**.
