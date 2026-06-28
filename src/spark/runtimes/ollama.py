"""Ollama backend — convenience daemon with its own model store.

Ollama serves all models from one long-running daemon. spark attaches to that
daemon (it never spawns or kills the shared daemon) and ensures the requested tag
is pulled before use. The OpenAI-compatible endpoint is at ``:11434/v1``."""

from __future__ import annotations

import json
import subprocess
import urllib.request

from ..config.schema import ModelEntry
from ..errors import LaunchError, RuntimeBackendError
from ..telemetry import get_telemetry
from .base import Backend, register


@register("ollama")
class OllamaBackend(Backend):
    def _tag(self, entry: ModelEntry) -> str:
        # Ollama addresses models by tag (e.g. 'llama3.1:8b').
        return entry.launch_overrides.get("ollama_tag") or entry.path or entry.id

    def _daemon_healthy(self, host: str, port: int) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/api/tags", timeout=3
            ) as r:
                return r.status == 200
        except Exception:
            return False

    def _installed_tags(self) -> list[str]:
        try:
            out = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=10,
                check=False,
            )
            tags = []
            for line in out.stdout.splitlines()[1:]:  # skip header
                parts = line.split()
                if parts:
                    tags.append(parts[0])
            return tags
        except (OSError, subprocess.SubprocessError):
            return []

    def prepare(self, entry: ModelEntry, *, secrets=None) -> None:
        host = self.config.general.host
        port = self.rt.server.default_port
        log = get_telemetry()

        if not self._daemon_healthy(host, port):
            raise LaunchError(
                "Ollama daemon is not running.",
                code="RT_UNAVAILABLE",
                remediation=[
                    "Start it: ollama serve   (or open the Ollama app)",
                    f"Verify: curl -s http://{host}:{port}/api/tags",
                ],
            )

        tag = self._tag(entry)
        if tag not in self._installed_tags():
            log.info("ollama", "pull_start", tag=tag)
            try:
                subprocess.run(["ollama", "pull", tag], check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeBackendError(
                    f"`ollama pull {tag}` failed (exit {exc.returncode}).",
                    remediation=[
                        f"Check the tag exists: ollama pull {tag}",
                        "See https://ollama.com/library for valid tags.",
                    ],
                    cause=exc,
                ) from exc
            log.info("ollama", "pull_done", tag=tag)
