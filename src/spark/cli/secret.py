"""`spark secret ...` — manage secrets in the macOS Keychain.

Values are read via getpass/stdin (never argv), shown only with an explicit
``--reveal``, and never logged."""

from __future__ import annotations

import getpass
import sys

import click

from ..errors import SecretError
from .context import build_context
from .render import console


@click.group(name="secret")
def secret_group():
    """Manage secrets (macOS Keychain backend)."""


@secret_group.command(name="set")
@click.argument("name")
@click.option("--stdin", "from_stdin", is_flag=True,
              help="Read the value from stdin instead of an interactive prompt.")
def secret_set(name: str, from_stdin: bool):
    """Store/overwrite a secret. Value is never taken from argv."""
    ctx = build_context()
    if from_stdin:
        value = sys.stdin.readline().rstrip("\n")
    else:
        value = getpass.getpass(f"value for '{name}' (input hidden): ")
    if not value:
        raise SecretError(
            "No value provided.",
            remediation=["Pipe a value with --stdin, or type it at the prompt."],
        )
    ctx.secrets.set(name, value)
    # value goes out of scope; it was registered for redaction inside the store.
    ctx.telemetry.info("cli", "secret_set", name=name)
    console.print(f"[green]✓[/green] stored secret [cyan]{name}[/cyan] in "
                  f"{ctx.secrets.backend_name}")


@secret_group.command(name="ls")
def secret_ls():
    """List secret names (never values)."""
    ctx = build_context()
    names = ctx.secrets.list_names()
    if not names:
        console.print("[dim]no secrets stored[/dim]")
        return
    for n in names:
        console.print(f"  • {n}")


@secret_group.command(name="rm")
@click.argument("name")
def secret_rm(name: str):
    """Remove a secret."""
    ctx = build_context()
    ctx.secrets.delete(name)
    ctx.telemetry.info("cli", "secret_rm", name=name)
    console.print(f"[green]✓[/green] removed [cyan]{name}[/cyan]")


@secret_group.command(name="get")
@click.argument("name")
@click.option("--reveal", is_flag=True, help="Print the secret value to stdout.")
def secret_get(name: str, reveal: bool):
    """Check a secret exists; print its value only with --reveal."""
    ctx = build_context()
    if not reveal:
        exists = ctx.secrets.exists(name)
        console.print(
            f"[cyan]{name}[/cyan]: "
            + ("[green]set[/green]" if exists else "[red]not set[/red]")
        )
        return
    value = ctx.secrets.get(name)
    # Deliberately bypass rich (no markup parsing of secret) and do not log.
    sys.stdout.write(value + "\n")
