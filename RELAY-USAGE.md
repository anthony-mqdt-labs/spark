# Spark Relay Backend — Quick Reference

The relay backend lets spark proxy requests to external OpenAI-compatible inference services (like the patchwork dynamic router, a remote inference service, or any other compatible endpoint).

## Quick Start (3 steps)

### 1. Register a relay model

```bash
cat > ~/.local/share/spark/models/my-external-service.toml << 'EOF'
id = "my-external-service"
description = "External inference service (via relay)"
model_format = "any"
backend = "relay"
research_status = "registered"

[server]
relay_base_url = "http://localhost:8000"        # Your external service
relay_health_endpoint = "/health"                 # Health check path
EOF
```

Or for the patchwork dynamic router specifically:

```bash
cat > ~/.local/share/spark/models/patchwork-router.toml << 'EOF'
id = "patchwork-router"
description = "Patchwork dynamic router (T0/T1/T2 cascade)"
model_format = "any"
backend = "relay"
research_status = "registered"

[server]
relay_base_url = "http://127.0.0.1:8000"
relay_health_endpoint = "/health"
EOF
```

Verify it was registered:
```bash
spark list | grep router
```

### 2. Start the external service

Example (patchwork router):
```bash
cd "$DARKCORE_ROUTER_DIR"   # the router project's root on your machine
MLXPY=$(uv tool dir)/mlx-lm/bin/python
$MLXPY -m darkcore.server --port 8000
```

### 3. Start spark attached to the relay

```bash
spark patchwork-router
```

Expected output:
```
Patchwork Dynamic Router — external service relay
  Health check: GET http://127.0.0.1:8000/health ✓
  Base URL: http://127.0.0.1:8000/v1
  Attaching to external service (never spawning local process)
```

## Usage

Your agent harness calls spark normally — it doesn't know about the relay:

```bash
# From agent harness code (Python, curl, or any OpenAI client):
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "patchwork-dynamic-router",
    "messages": [{"role": "user", "content": "What is a neural network?"}]
  }' | python3 -m json.tool
```

Spark handles the relay transparently:
```
Agent harness → spark (8080) → relay backend → external service (8000)
```

## Configuration

The relay backend reads two config fields:

| Field | Default | Notes |
|-------|---------|-------|
| `relay_base_url` | "" | The external service's base URL (e.g., `http://localhost:8000`) |
| `relay_health_endpoint` | "/health" | Path for health checks (appended to base_url) |

Both are set in the model's TOML file under `[server]`:

```toml
[server]
relay_base_url = "https://api.example.com"
relay_health_endpoint = "/v1/models"  # Alternative: check models endpoint
```

## Monitoring

Spark logs all relay activity to `~/.local/share/spark/logs/`:

```bash
# Watch relay health checks
tail -f ~/.local/share/spark/logs/supervisor/$(date +%Y-%m-%d).jsonl | grep relay

# Check request proxying
tail -f ~/.local/share/spark/logs/supervisor/$(date +%Y-%m-%d).jsonl | grep "v1/chat"
```

## Troubleshooting

### "Health check failed"

```
Error: Unable to reach http://127.0.0.1:8000/health
```

- Verify the external service is running on that port
- Check the `relay_base_url` in your model config
- Test manually: `curl http://127.0.0.1:8000/health`

### "Connection refused"

The relay can't reach the external service. Common causes:
- External service crashed or was shut down
- Wrong port in `relay_base_url`
- Firewall blocking (if service is remote)

### Slow responses

- Check if the external service itself is slow (query it directly)
- Check spark's logs: `tail -f ~/.local/share/spark/logs/supervisor/*.jsonl | jq '.latency_ms'`

## Multiple External Services

You can register multiple relay models pointing to different services:

```bash
# Router
cat > ~/.local/share/spark/models/router.toml << 'EOF'
id = "router"
backend = "relay"
research_status = "registered"
[server]
relay_base_url = "http://localhost:8000"
EOF

# Remote API
cat > ~/.local/share/spark/models/remote-api.toml << 'EOF'
id = "remote-api"
backend = "relay"
research_status = "registered"
[server]
relay_base_url = "https://api.example.com"
EOF

# Local dev server
cat > ~/.local/share/spark/models/dev.toml << 'EOF'
id = "dev"
backend = "relay"
research_status = "registered"
[server]
relay_base_url = "http://localhost:3000"
EOF
```

Then switch between them:
```bash
spark router          # The patchwork router
spark remote-api      # The remote service
spark dev             # Your local dev server
```

## See Also

- **Patchwork router setup:** `../patchwork/experiments/router/ROUTER-SERVER.md`
- **Spark architecture:** `../spark/README.md` (§8 Subsystems in depth)
- **Backend configuration:** `../spark/config/runtimes/relay.toml` (example)
