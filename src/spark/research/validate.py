"""Accept-time validation for research output: trust, but verify against disk.

Staged research is an LLM's claims about a runtime's CLI surface — and LLMs
confabulate flags (observed: ``--max-kv-size`` recommended for ``mlx_lm``,
which only exists on ``mlx_vlm``; the server died on argparse, three restarts,
then a terminal failure). The registry entry is also where a non-canonical
quant tag (``2bit``) silently disables the memory budget, which only knows
``qN``-style tags.

So before staged output (or a manual import) touches an entry:

* ``normalize_quant`` canonicalizes ``Nbit``/``N-bit`` → ``qN``. GGUF-style tags
  (``q8_0``, ``Q4_K_M``) pass through untouched — they are validated by their
  own runtime, not by us.
* ``validate_flags`` checks every override the entry *would* carry after the
  merge (staged rec ⊕ pre-existing entry overrides — a backend switch must not
  smuggle the old server's flags along) against the recommended backend's real
  ``--help`` surface, probed live from the installed binary.

Probing can fail (binary absent, hangs — observed: llama.cpp 0.4.1 compiles
Metal on first exec and ignores ``--version`` entirely). An unprovable surface
is a *warning*, never a refusal: the operator's explicit ``--accept`` outranks
a check spark could not run. A provably-absent flag is a refusal with the fix
(re-research, hand-edit, or drop the flag).
"""

from __future__ import annotations

import re
import subprocess

from ..config.schema import ModelEntry, SparkConfig
from .staging import apply_output_to_entry
from .types import ResearchOutput

#: `4bit`, `4-bit`, `4 bit` (any case) -> `q4`. Only the bare Nx-bit shape is
#: rewritten; anything else (q8_0, Q4_K_M, f16, …) is left byte-identical.
_BIT_RE = re.compile(r"^([2-8])\s*-?bit$", re.I)

#: A CLI flag token: `--max-tokens`, `--jinja`. Single-dash tokens are never
#: valid overrides (D9: overrides are appended as real CLI flags).
_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")


def normalize_quant(tag: str) -> str:
    """Canonicalize a quant tag. Returns the tag unchanged unless it is the
    bare ``Nbit`` shape the budget cannot price."""
    t = (tag or "").strip()
    m = _BIT_RE.match(t)
    return f"q{m.group(1)}" if m else t


def known_flags(binary: str, *, timeout_s: float = 15.0) -> set[str] | None:
    """The ``--flag`` surface of an installed runtime binary, or None when it
    cannot be determined (missing binary, probe timeout, unparsable output).

    Never raises: an unprovable surface is a warning at the call site, not an
    error here.
    """
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output=True, text=True, timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    found = set(_FLAG_RE.findall(text))
    return found or None


def validate_flags(
    overrides: dict[str, str], backend: str, config: SparkConfig
) -> tuple[list[str], list[str]]:
    """(errors, warnings) for one backend's merged override set.

    Errors name flags the installed binary provably lacks. Warnings cover what
    could not be checked: unknown backend, unresolvable binary, or a surface
    probe that failed (llama.cpp 0.4.1 precedent: first exec compiles Metal and
    never answers ``--help`` in time).
    """
    errors: list[str] = []
    warnings: list[str] = []
    rt = config.runtimes.get(backend) if backend else None
    if rt is None:
        return (
            [f"Recommended backend '{backend}' is not a known runtime."],
            [],
        )
    surface = known_flags(rt.server.binary)
    if surface is None:
        warnings.append(
            f"Could not probe `{rt.server.binary} --help` — flag checks skipped "
            f"(accept proceeds on your say-so)."
        )
        return errors, warnings
    for flag in sorted(overrides):
        if not flag.startswith("--") or flag not in surface:
            errors.append(
                f"`{flag}` is not a flag of `{rt.server.binary}` "
                f"(per its installed --help)."
            )
    return errors, warnings


def validate_accept(
    entry: ModelEntry, output: ResearchOutput, config: SparkConfig
) -> tuple[ModelEntry, list[str], list[str]]:
    """Apply ``output`` to a copy of ``entry`` with normalization + flag checks.

    Returns ``(candidate, errors, warnings)``. The caller refuses to save when
    ``errors`` is non-empty; ``warnings`` are printed and do not block. ``entry``
    itself is untouched so a refusal leaves the registry exactly as it was.
    """
    candidate = entry.model_copy(deep=True)
    notes: list[str] = []
    rec = output.recommended()
    if rec and rec.quant:
        fixed = normalize_quant(rec.quant)
        if fixed != rec.quant:
            notes.append(f"quant {rec.quant!r} → {fixed!r}")
        rec = rec.model_copy(update={"quant": fixed})
        output = output.model_copy(
            update={
                "per_runtime": {**output.per_runtime, output.recommended_backend: rec}
            }
        )
    apply_output_to_entry(candidate, output)
    backend = candidate.backend
    errors, warnings = validate_flags(candidate.launch_overrides, backend, config)
    return candidate, errors, [*notes, *warnings]


__all__ = [
    "known_flags",
    "normalize_quant",
    "validate_accept",
    "validate_flags",
]
