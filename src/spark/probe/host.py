"""Host capability detection + caching.

Detects chip / unified memory / cores, and for each configured runtime whether it
is present, its version, and resolved path. Results are cached to the host profile
TOML with a content hash + timestamp; ``spark doctor`` forces a re-probe.

Detection is read-only and tolerant: a missing/uncooperative runtime is recorded
as unavailable with a reason, never an exception.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config.schema import RuntimeDef, SparkConfig
from ..telemetry import get_telemetry


@dataclass
class RuntimeStatus:
    name: str
    available: bool
    reason: str = ""           # why unavailable (if so)
    path: str = ""             # resolved binary path
    version: str = ""          # best-effort version string
    install_hint: str = ""


@dataclass
class HostProfile:
    os: str
    arch: str
    chip: str
    total_memory_bytes: int
    cpu_count: int
    probed_at: float
    runtimes: dict[str, RuntimeStatus] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def available_runtimes(self) -> list[str]:
        return [n for n, s in self.runtimes.items() if s.available]


# --- low-level host facts ------------------------------------------------------
def _sysctl(key: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _total_memory_bytes() -> int:
    if platform.system() == "Darwin":
        val = _sysctl("hw.memsize")
        if val.isdigit():
            return int(val)
    try:  # portable fallback
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _chip() -> str:
    if platform.system() == "Darwin":
        brand = _sysctl("machdep.cpu.brand_string")
        if brand:
            return brand
    return platform.processor() or platform.machine()


# --- runtime detection ---------------------------------------------------------
_VERSION_RE = re.compile(r"\b\d+(?:\.\d+){1,3}\b|\bversion:?\s*\S+", re.I)


def _detect_version(binary_path: str, args: list[str], timeout: float) -> str:
    """Best-effort version string. Many runtimes (mlx_*) lack --version and dump a
    usage banner instead — we extract a version-looking token or return ''."""
    try:
        out = subprocess.run(
            [binary_path, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        blob = f"{out.stdout}\n{out.stderr}"
        for line in blob.splitlines():
            line = line.strip()
            if line.lower().startswith(("usage:", "warning:", "error:")):
                continue
            m = _VERSION_RE.search(line)
            if m:
                return line[:80]
        return ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_runtime(rt: RuntimeDef, *, os_name: str, arch: str) -> RuntimeStatus:
    # Host gating first — cheaper and gives a precise reason.
    if rt.requires_os and os_name not in rt.requires_os:
        return RuntimeStatus(
            name=rt.name, available=False,
            reason=rt.unavailable_reason or f"requires OS {rt.requires_os}",
            install_hint=rt.install_hint,
        )
    if rt.requires_arch and arch not in rt.requires_arch:
        return RuntimeStatus(
            name=rt.name, available=False,
            reason=rt.unavailable_reason or f"requires arch {rt.requires_arch}",
            install_hint=rt.install_hint,
        )
    path = shutil.which(rt.detect.binary)
    if not path:
        return RuntimeStatus(
            name=rt.name, available=False,
            reason=f"'{rt.detect.binary}' not found on PATH",
            install_hint=rt.install_hint,
        )
    version = _detect_version(path, rt.detect.version_args, rt.detect.version_timeout_s)
    return RuntimeStatus(
        name=rt.name, available=True, path=path, version=version,
        install_hint=rt.install_hint,
    )


def _interpreter_for(binary: str) -> str | None:
    """Resolve the Python interpreter backing a console-script `binary` by reading
    its shebang. Returns None if it can't be determined (e.g. a compiled binary)."""
    path = shutil.which(binary)
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            first = fh.readline(256)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    parts = first[2:].strip().decode("utf-8", "replace").split()
    if not parts:
        return None
    # "#!/usr/bin/env python3.12" -> resolve the named interpreter; otherwise the
    # shebang is a direct path to the venv interpreter (the uv-tool / pip case).
    if parts[0].rsplit("/", 1)[-1] == "env" and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0]


def missing_python_deps(binary: str, modules: list[str]) -> list[str]:
    """Subset of `modules` NOT importable by `binary`'s own interpreter.

    Uses importlib.find_spec (no heavy import) in the runtime's venv, so it reports
    exactly what that runtime would see at inference time. Returns [] when there is
    nothing to check or the interpreter can't be resolved (unknown, not 'missing')."""
    if not modules:
        return []
    interp = _interpreter_for(binary)
    if not interp:
        return []
    code = (
        "import importlib.util,sys;"
        "print('\\n'.join(m for m in sys.argv[1:] "
        "if importlib.util.find_spec(m) is None))"
    )
    try:
        out = subprocess.run(
            [interp, "-c", code, *modules],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [m for m in out.stdout.split() if m]


def probe_host(config: SparkConfig) -> HostProfile:
    os_name = platform.system()
    arch = platform.machine()
    profile = HostProfile(
        os=os_name,
        arch=arch,
        chip=_chip(),
        total_memory_bytes=_total_memory_bytes(),
        cpu_count=_cpu_count(),
        probed_at=time.time(),
    )
    log = get_telemetry()
    for name, rt in config.runtimes.items():
        status = detect_runtime(rt, os_name=os_name, arch=arch)
        profile.runtimes[name] = status
        log.info(
            "probe", "runtime_detected",
            runtime=name, available=status.available,
            version=status.version, reason=status.reason,
        )
    log.info(
        "probe", "host_profiled",
        chip=profile.chip, arch=arch,
        total_memory_gib=round(profile.total_memory_bytes / 2**30, 1),
        available=profile.available_runtimes,
    )
    return profile


def _cpu_count() -> int:
    import os
    return os.cpu_count() or 0


# --- caching -------------------------------------------------------------------
def _config_fingerprint(config: SparkConfig) -> str:
    blob = json.dumps(
        {n: rt.model_dump() for n, rt in sorted(config.runtimes.items())},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_or_probe(
    config: SparkConfig, cache_path: Path, *, max_age_s: float = 86400.0,
    force: bool = False,
) -> HostProfile:
    """Return a cached profile if fresh and config-consistent, else re-probe."""
    fp = _config_fingerprint(config)
    if not force and cache_path.exists():
        try:
            import tomllib
            with cache_path.open("rb") as fh:
                cached = tomllib.load(fh)
            if (
                cached.get("_fingerprint") == fp
                and (time.time() - cached.get("probed_at", 0)) < max_age_s
            ):
                return _profile_from_cache(cached)
        except Exception:
            pass  # fall through to re-probe
    profile = probe_host(config)
    _write_cache(profile, cache_path, fp)
    return profile


def _profile_from_cache(data: dict) -> HostProfile:
    runtimes = {
        n: RuntimeStatus(**s) for n, s in data.get("runtimes", {}).items()
    }
    return HostProfile(
        os=data["os"], arch=data["arch"], chip=data["chip"],
        total_memory_bytes=data["total_memory_bytes"],
        cpu_count=data["cpu_count"], probed_at=data["probed_at"],
        runtimes=runtimes,
    )


def _write_cache(profile: HostProfile, cache_path: Path, fingerprint: str) -> None:
    # Hand-serialize a small TOML (avoids adding a TOML *writer* dependency).
    lines = [
        "# spark host profile cache (auto-generated; safe to delete).",
        f'os = "{profile.os}"',
        f'arch = "{profile.arch}"',
        f'chip = "{profile.chip}"',
        f"total_memory_bytes = {profile.total_memory_bytes}",
        f"cpu_count = {profile.cpu_count}",
        f"probed_at = {profile.probed_at}",
        f'_fingerprint = "{fingerprint}"',
        "",
    ]
    for name, s in profile.runtimes.items():
        lines.append(f"[runtimes.{name}]")
        lines.append(f'name = "{s.name}"')
        lines.append(f"available = {str(s.available).lower()}")
        lines.append(f'reason = {_toml_str(s.reason)}')
        lines.append(f'path = {_toml_str(s.path)}')
        lines.append(f'version = {_toml_str(s.version)}')
        lines.append(f'install_hint = {_toml_str(s.install_hint)}')
        lines.append("")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(lines), encoding="utf-8")


def _toml_str(s: str) -> str:
    """Encode a string as a TOML basic string literal safely."""
    return json.dumps(s if s is not None else "")
