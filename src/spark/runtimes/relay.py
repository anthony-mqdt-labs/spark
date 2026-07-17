"""Relay backend — proxy requests to external OpenAI-compatible services.

The relay backend forwards inference requests to an external HTTP endpoint
(e.g., patchwork dynamic router, another spark instance, a deployed inference
service). This is transparent to the agent harness: spark attaches to the
external service without spawning a local process."""

from __future__ import annotations

import urllib.request

from ..config.schema import ModelEntry
from ..errors import LaunchError
from ..telemetry import get_telemetry
from .base import Backend, register


@register("relay")
class RelayBackend(Backend):
    """Proxy backend that forwards requests to an external OpenAI-compatible
    endpoint.

    Instead of spawning a local inference server, this backend:
    - Attaches to an already-running external service
    - Proxies health checks to the external endpoint
    - Routes OpenAI-compatible requests to the external service
    """

    def build_launch_cmd(
        self, entry: ModelEntry, host: str, port: int
    ) -> list[str]:
        """Relay backend does not spawn a process.

        Returns an empty list; the external service is assumed to be already
        running. The supervisor will detect this is a relay via attach_if_running=True.
        """
        return []

    def health_url(self, host: str, port: int) -> str:
        """Health check URL points to the external relay service.

        Override the default health_url to use the configured relay endpoint
        instead of localhost:port.
        """
        base_url = self.rt.server.relay_base_url
        health_path = self.rt.server.relay_health_endpoint
        if not health_path.startswith("/"):
            health_path = "/" + health_path
        return f"{base_url}{health_path}"

    def openai_base_url(self, host: str, port: int) -> str:
        """OpenAI-compatible base URL points to the external relay service.

        Clients should point their OpenAI SDK at this URL to forward requests
        through the relay.
        """
        base_url = self.rt.server.relay_base_url
        if self.openai_compatible:
            return f"{base_url}/v1"
        return base_url

    def prepare(self, entry: ModelEntry, *, secrets=None) -> None:
        """Pre-flight check: verify the external relay service is reachable.

        Logs a warning if the endpoint appears unreachable, but does not raise
        (allows launch to proceed; supervisor will detect health failure).
        """
        log = get_telemetry()
        base_url = self.rt.server.relay_base_url

        if not base_url:
            raise LaunchError(
                "Relay backend requires relay_base_url in configuration.",
                code="CFG_INVALID",
                remediation=[
                    "Set relay_base_url in your spark config.",
                    "Example: relay_base_url = 'http://localhost:8000'",
                ],
            )

        # Soft health check: warn if unreachable, but don't fail launch
        try:
            health_url = self.health_url("", 0)  # host/port ignored for relay
            with urllib.request.urlopen(health_url, timeout=3) as r:
                if r.status == 200:
                    log.info("relay", "external_service_healthy", url=base_url)
                    return
        except Exception as exc:
            log.warn(
                "relay",
                "external_service_unreachable",
                url=base_url,
                reason=str(exc),
            )
            # Don't raise: let supervisor discover the failure via health checks
