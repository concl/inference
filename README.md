# inference

## Serving

`scripts/serve.py` is the definition of how the model is served, including the
option list.

```bash
inference serve --help          # option list
inference serve                 # defaults
inference serve --port 8080     # any option as a flag
PORT=8080 inference serve       # or as an environment variable

# long-lived, detached
screen -S vllm_server -dm inference serve
```

Flags take precedence over environment variables. Run it under `screen` or
`systemd` so it outlives your SSH session. `python -m scripts.serve` also works.

## Client

```bash
inference check                          # one prompt
inference check --interactive            # chat
inference check --model Qwen3.8-27B --prompt "hi"
```