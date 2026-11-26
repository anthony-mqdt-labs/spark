"""Shell completion: the `_spark` zsh function + hidden `__complete` data source.

`spark completion zsh` prints a native compsys function (zero runtime deps). It is
fzf-tab compatible: when fzf-tab is installed, the same candidates render as a
fuzzy popup, and the documented preview zstyle calls `spark __complete describe`.

The candidate list is availability-driven and context-aware: a launch context
(`run`, or a bare `spark <TAB>`) never offers an entry whose weights are provably
gone, while administrative contexts (`forget`, `research`, `download`) still name
it — marked — because `spark forget` is how such an entry is removed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from ..config.paths import resolve_paths
from ..registry import list_models, resolve_model

if TYPE_CHECKING:
    from ..config.paths import SparkPaths
    from ..config.schema import ModelEntry
    from ..inventory import Inventory
    from ..registry.availability import Availability

#: Rider appended to a candidate whose weights are absent.
MISSING_MARK = "MISSING WEIGHTS"

#: Contexts whose argument must be a model this host can actually launch.
LAUNCH_CONTEXTS = frozenset({"", "run"})

#: Short-lived, *in-process* preview cache for unregistered-but-present weights.
#: fzf-tab launches a new process per keystroke, so this cache cannot span
#: keystrokes — it only helps callers that ask for several previews in one
#: process. The effective key is (id, root mtimes, resolved location, that
#: location's st_mtime_ns): a snapshot swapped in place invalidates it even when
#: the roots' own mtimes did not change. Value: (text, location, location mtime).
_DESCRIBE_CACHE: dict[tuple[str, tuple[float, ...]], tuple[str, str, int]] = {}
_DESCRIBE_CACHE_MAX = 64


def _location_mtime_ns(location: str) -> int:
    """Freshness stamp for one resolved snapshot/store directory."""
    try:
        return Path(location).stat().st_mtime_ns
    except OSError:
        return 0


def _discover_signature(paths: SparkPaths) -> tuple[float, ...]:
    """Cheap freshness key for the inputs an inventory is built from."""
    from ..hf_cache import hf_cache_root

    sig: list[float] = []
    for root in (paths.models_dir, paths.model_store_dir, hf_cache_root()):
        try:
            sig.append(root.stat().st_mtime)
        except OSError:
            sig.append(0.0)
    return tuple(sig)


def _registry_entries(inv: Inventory) -> list[tuple[ModelEntry, Availability]]:
    """Every registry entry with the availability already computed by ``inv``.

    ``build_inventory`` has just read every TOML and resolved availability;
    re-listing and re-resolving here would double that work on every Tab.
    """
    entries = (*inv.runnable_registered, *inv.external, *inv.unverifiable, *inv.missing)
    return sorted(entries, key=lambda pair: pair[0].id.lower())


# Native zsh completion. fzf-tab, if present, upgrades this to a fuzzy menu.
_ZSH_COMPLETION = r"""#compdef spark
# spark zsh completion (native compsys; fzf-tab compatible).
# Install: spark completion zsh > "${fpath[1]}/_spark" && compinit
# Fuzzy preview (with fzf-tab):
#   zstyle ':fzf-tab:complete:spark:*' fzf-preview 'spark __complete describe $word'

_spark() {
  local -a subcmds
  subcmds=(
      'run:Launch the optimal runtime for a model'
      'download:Download + register a Hugging Face model'
      'adopt:Register an on-disk model into the registry'
      'research:Research optimal config for a registered model'
      'doctor:Probe host capabilities and runtimes'
      'list:List what this host can serve'
      'catalog:Print the live model catalog'
      'forget:Remove a model from the registry'
      'secret:Manage secrets (macOS Keychain)'
      'config:Inspect and validate configuration'
      'completion:Emit shell completion'
  )

  # Availability decides the menu (computed live at Tab time): a launch context
  # must not offer a model whose weights are gone, while `forget` must still
  # name it so it can be removed. At position 2 the word being completed is the
  # subcommand itself, so the context stays empty (launch) there.
  local ctx=""
  (( CURRENT > 2 )) && ctx="${words[2]}"

  local -a models
  local id desc
  while IFS=$'\t' read -r id desc; do
    [[ -n "$id" ]] && models+=("${id}:${desc}")
  done < <(command spark __complete models "$ctx" 2>/dev/null)

  if (( CURRENT == 2 )); then
    _describe -t commands 'spark command' subcmds
    _describe -t models 'model' models
    return
  fi

  case "${words[2]}" in
    run|download|research|forget|adopt)
      _describe -t models 'model' models ;;
    secret)
      _values 'secret command' set ls rm get ;;
    config)
      _values 'config command' validate path show review import ;;
    completion)
      _values 'shell' zsh ;;
  esac
}

_spark "$@"
"""


@click.command(name="completion")
@click.argument("shell", type=click.Choice(["zsh"]), default="zsh")
def completion_command(shell: str) -> None:
    """Print the shell completion script (currently: zsh)."""
    sys.stdout.write(_ZSH_COMPLETION)


def completion_models(
    paths: SparkPaths,
    context: str = "",
    *,
    inventory: Inventory | None = None,
) -> list[tuple[str, str]]:
    """(model id, description) candidates for one completion context.

    The menu is disk truth: every runnable model — registered or merely present
    — is offered in launch contexts. Availability is computed without walking
    weights (`with_bytes=False`) and without loading config: this path must stay
    fast and offline (README §9.3). Missing registry entries stay nameable in
    administrative contexts (marked), because naming them is how they get
    cleaned up; unregistered disk models are offered only where they can run.

    ``inventory`` lets a caller that already scanned disk pass that result in
    instead of paying for a second build.
    """
    from ..inventory import build_inventory

    offer_missing = context.strip().lower() not in LAUNCH_CONTEXTS
    is_launch = context.strip().lower() in LAUNCH_CONTEXTS
    candidates: list[tuple[str, str]] = []
    try:
        inv = (
            inventory
            if inventory is not None
            else build_inventory(paths, config=None, with_bytes=False)
        )
    except Exception:
        inv = None
    if inv is not None:
        if is_launch:
            for entry, _a in (*inv.runnable_registered, *inv.external, *inv.unverifiable):
                desc = (
                    " · ".join(x for x in (entry.backend, entry.quant, entry.model_format) if x)
                    or "model"
                )
                candidates.append((entry.id, desc))
            for m in inv.runnable_discovered:
                candidates.append((m.display_id, f"{m.source} · on disk · not registered"))
            return candidates
        # Administrative contexts: registry entries (missing marked). `forget`
        # can only drop a registry entry, so it names registry entries alone;
        # research/download also name discovered disk models (adoptable, or
        # directly runnable).
        from ..registry.availability import MISSING

        if context.strip().lower() != "forget":
            for m in inv.runnable_discovered:
                candidates.append((m.display_id, f"{m.source} · on disk · not registered"))

        for mod, avail in _registry_entries(inv):
            desc = " · ".join(x for x in (mod.backend, mod.quant, mod.model_format) if x) or "model"
            if avail.state == MISSING:
                desc = f"{desc} · {MISSING_MARK}"
            candidates.append((mod.id, desc))
        return candidates

    from ..registry.availability import MISSING, resolve_availability

    for m in list_models(paths):
        desc = " · ".join(x for x in (m.backend, m.quant, m.model_format) if x) or "model"
        a = resolve_availability(m, paths, with_bytes=False)
        if a.state == MISSING:
            if not offer_missing:
                continue
            desc = f"{desc} · {MISSING_MARK}"
        candidates.append((m.id, desc))
    return candidates


def _describe_discovered(arg: str, paths: SparkPaths) -> str:
    """Preview for unregistered-but-present weights, without a full byte walk.

    The inventory is built with ``with_bytes=False`` (registry + dirs only); the
    size is measured for this one location instead of every cached snapshot.
    A cached preview is checked first, using the id, input mtimes and the
    resolved location's ``st_mtime_ns``, so a repeat skips both the inventory
    build and the size walk — but an in-place snapshot swap is never stale.
    """
    from ..hf_cache import bytes_in
    from ..inventory import build_inventory

    key = (arg.strip().lower(), _discover_signature(paths))
    cached = _DESCRIBE_CACHE.get(key)
    if cached is not None:
        text, location, cached_mtime = cached
        # The location is known only after a scan; validating its mtime here is
        # what puts it in the effective cache key (a miss has no location).
        if not location or _location_mtime_ns(location) == cached_mtime:
            return text
    inv = build_inventory(paths, config=None, with_bytes=False)
    hit = next(
        (d for d in inv.runnable_discovered if d.display_id.lower() == arg.lower()),
        None,
    )
    if hit is None:
        # A miss is cached too: typing an id that is neither registered nor on
        # disk must not rescan the cache on every remaining keystroke.
        if len(_DESCRIBE_CACHE) >= _DESCRIBE_CACHE_MAX:
            _DESCRIBE_CACHE.clear()
        _DESCRIBE_CACHE[key] = ("", "", 0)
        return ""
    present = hit.bytes_present or bytes_in(Path(hit.location))
    size = f"{present / 2**30:.1f} GiB" if present else "—"
    text = (
        f"id:       {hit.display_id}\n"
        f"source:   {hit.source} (on disk, not registered)\n"
        f"format:   {hit.model_format}\n"
        f"weights:  {hit.kind} · {hit.location} · {size}\n"
        f"adopt:    spark adopt {hit.display_id}\n"
    )
    if len(_DESCRIBE_CACHE) >= _DESCRIBE_CACHE_MAX:
        _DESCRIBE_CACHE.clear()
    _DESCRIBE_CACHE[key] = (text, hit.location, _location_mtime_ns(hit.location))
    return text


def _describe(arg: str, paths: SparkPaths) -> None:
    from ..registry.availability import resolve_availability

    try:
        m = resolve_model(arg, paths)
    except Exception:
        m = None
    if m is None:
        sys.stdout.write(_describe_discovered(arg, paths))
        return
    a = resolve_availability(m, paths)
    lines = [
        f"id:       {m.id}",
        f"backend:  {m.backend or '—'}",
        f"format:   {m.model_format}",
        f"quant:    {m.quant or '—'}",
        f"repo:     {m.hf_repo or '—'}",
        f"weights:  {a.describe()}",
        f"status:   {m.research_status}",
    ]
    sys.stdout.write("\n".join(lines) + "\n")


@click.command(name="__complete", hidden=True)
@click.argument("what")
@click.argument("arg", required=False)
def complete_command(what: str, arg: str | None) -> None:
    """Hidden machine-readable completion data. Never logs; tolerant of errors.

    `what=models` takes an optional context — the subcommand the model is being
    chosen for — which decides whether entries without weights are offered.
    """
    try:
        paths = resolve_paths()
        if what == "models":
            for model_id, desc in completion_models(paths, arg or ""):
                sys.stdout.write(f"{model_id}\t{desc}\n")
        elif what == "describe" and arg:
            _describe(arg, paths)
    except Exception:
        # Completion must never surface errors into the shell.
        return
