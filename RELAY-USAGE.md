# Spark relay backend — quick reference

The `relay` runtime lets spark **attach to** an external OpenAI-compatible
service, such as the patchwork dynamic router or another inference server that
is already running. spark health-checks it and publishes its URL as the
endpoint, so consumers that read spark's catalog or instance record find it.

What relay does **not** do:
- **No spawning or killing.** The external service is yours to start and stop.
- **No proxying.** Consumers talk to the external URL directly. spark sits only
  on the discovery path, not the request path.
- **No auth.** If the service needs an API key, use a hosted-provider runtime
  instead (`openrouter`/`deepseek`, or a new TOML with `provider_base_url`).
  It runs a loopback proxy that holds the key. See README §3, "Use a hosted
  provider".

(An earlier version of this file described a spark-side proxy on `:8080` and a
`[server]` table inside model entries. Neither exists: a model entry carrying
`[server]` is rejected as invalid and skipped by `spark list`.)

## Setup (3 steps)

### 1. Point the relay runtime at your service

The URL belongs to the **runtime**, not the model. Override it in your user
config (`spark config path` shows where that file lives, usually
`~/.config/spark/config.toml`):

```toml
[runtimes.relay.server]
relay_base_url = "http://127.0.0.1:8000"   # your service's base URL (no /v1)
relay_health_endpoint = "/health"           # GET path; must answer 2xx
```

This deep-merges over the bundled `config/runtimes/relay.toml`, so only these
two keys are needed. Check it with `spark config validate`.

### 2. Register a model entry that uses it

```bash
cat > ~/.local/share/spark/models/patchwork-router.toml <<'EOF'
id = "patchwork-router"
backend = "relay"
research_status = "manual"
EOF
spark list | grep patchwork-router   # shows src "daemon", backend "relay"
```

### 3. Start your service, then attach

```bash
# e.g. the patchwork router (see ../patchwork/experiments/router/ROUTER-SERVER.md)
spark patchwork-router
```

If the service is healthy, spark attaches. It prints the endpoint
(`http://127.0.0.1:8000/v1`), publishes `run/instances/patchwork-router.json`,
and stays attached until Ctrl+C, which never stops the service. If the service
is not healthy, the launch fails with `RUN_LAUNCH`, the probe URL, and what to
do next. Nothing is spawned.

## Using it

Consumers call the **external service's** URL, either read from the instance
record (`base_url`) or taken from what spark printed:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"patchwork-dynamic-router","messages":[{"role":"user","content":"hi"}]}'
```

## Limits

- **One relay target at a time.** Only the runtime named `relay` gets relay
  behaviour (attach, and health at the external URL). A second
  `[runtimes.<other>]` copy would launch as a generic backend and fail. To
  switch services, change `relay_base_url`. To front several hosted APIs, use
  provider runtimes, one TOML each.
- spark's supervision covers health only: no restart, because the process isn't
  spark's.

## Troubleshooting

- **`RUN_LAUNCH: 'relay' endpoint is not healthy`.** The service isn't
  answering 2xx at `relay_base_url + relay_health_endpoint`. Probe it with
  `curl -s <that URL>`, then check that the service is running and the port is
  right.
- **Entry missing from `spark list`.** Run `spark list --all`. A `skipped
  invalid registry entry` line means the model TOML has keys a model entry
  doesn't accept (for example a `[server]` table).
- **Logs.** `~/.local/share/spark/logs/supervisor/<date>.jsonl` has the
  `attached` / `detached` events. `logs/relay/<date>.jsonl` has the pre-launch
  reachability check (`external_service_healthy` / `external_service_unreachable`).
