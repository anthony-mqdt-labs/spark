"""Shell completion: the `_spark` zsh function + hidden `__complete` data source.

`spark completion zsh` prints a native compsys function (zero runtime deps). It is
fzf-tab compatible: when fzf-tab is installed, the same candidates render as a
fuzzy popup, and the documented preview zstyle calls `spark __complete describe`.
"""

from __future__ import annotations

import sys

import click

from ..config.paths import resolve_paths
from ..registry import list_models, resolve_model

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
      'research:Research optimal config for a registered model'
      'doctor:Probe host capabilities and runtimes'
      'list:List registered models'
      'forget:Remove a model from the registry'
      'secret:Manage secrets (macOS Keychain)'
      'config:Inspect and validate configuration'
      'completion:Emit shell completion'
  )

  local -a models
  local id desc
  while IFS=$'\t' read -r id desc; do
    [[ -n "$id" ]] && models+=("${id}:${desc}")
  done < <(command spark __complete models 2>/dev/null)

  if (( CURRENT == 2 )); then
    _describe -t commands 'spark command' subcmds
    _describe -t models 'model' models
    return
  fi

  case "${words[2]}" in
    run|download|research|forget)
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
def completion_command(shell: str):
    """Print the shell completion script (currently: zsh)."""
    sys.stdout.write(_ZSH_COMPLETION)


@click.command(name="__complete", hidden=True)
@click.argument("what")
@click.argument("arg", required=False)
def complete_command(what: str, arg: str | None):
    """Hidden machine-readable completion data. Never logs; tolerant of errors."""
    try:
        paths = resolve_paths()
        if what == "models":
            # Availability is computed without walking weights (`with_bytes=False`)
            # and without loading config: this path must stay fast and offline
            # (README §9.3). A MISSING marker here means the entry asserts weights
            # that are absent — the one signal a consumer must not offer as runnable.
            from ..registry.availability import MISSING, resolve_availability

            for m in list_models(paths):
                desc = " · ".join(
                    x for x in (m.backend, m.quant, m.model_format) if x
                ) or "model"
                a = resolve_availability(m, paths, with_bytes=False)
                if a.state == MISSING:
                    desc = f"{desc} · MISSING WEIGHTS"
                sys.stdout.write(f"{m.id}\t{desc}\n")
        elif what == "describe" and arg:
            from ..registry.availability import resolve_availability

            m = resolve_model(arg, paths)
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
    except Exception:
        # Completion must never surface errors into the shell.
        return

