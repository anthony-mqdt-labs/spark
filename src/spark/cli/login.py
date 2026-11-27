"""`spark login|logout <runtime>`: ChatGPT-plan sign-in (spec 0005).

Opens OpenAI's "Sign in with ChatGPT" page in the browser. The session
(issued client id + rotating tokens) lands in the macOS Keychain and is never
printed. Only runtimes with ``[server.provider_oauth]`` in their TOML qualify.
"""

from __future__ import annotations

import webbrowser

import click

from ..errors import ConfigError, SparkError
from .context import build_context
from .render import console, hint, success, warning


def _oauth_runtime(ctx, runtime: str):
    rt = ctx.config.runtimes.get(runtime)
    if rt is None or rt.server.provider_oauth is None:
        eligible = sorted(n for n, r in ctx.config.runtimes.items() if r.server.provider_oauth)
        raise ConfigError(
            f"'{runtime}' is not a ChatGPT sign-in provider.",
            code="CFG_NOT_FOUND",
            remediation=[f"Sign-in providers: {', '.join(eligible) or '(none configured)'}"],
            context={"runtime": runtime},
        )
    return rt


@click.command(name="login")
@click.argument("runtime")
@click.option(
    "--no-browser",
    is_flag=True,
    help="Print the sign-in URL instead of opening a browser.",
)
def login_command(runtime: str, no_browser: bool):
    """Sign in with ChatGPT so RUNTIME can spend your plan's usage."""
    from ..secrets.oauth import list_models, login, oauth_lock_path

    ctx = build_context()
    rt = _oauth_runtime(ctx, runtime)
    spec = rt.server.provider_oauth

    def announce(url: str) -> None:
        console.print("Opening the ChatGPT sign-in page. If no browser appears, open:")
        console.print(url, soft_wrap=True, markup=False, highlight=False)

    sess = login(
        ctx.secrets,
        spec,
        runtime,
        oauth_lock_path(ctx.paths, spec),
        open_browser=(lambda _url: None) if no_browser else webbrowser.open,
        announce=announce,
    )
    success(f"signed in; plan usage granted ({spec.required_scope}); session in the Keychain")
    try:
        models = list_models(rt.server.provider_base_url, sess.access_token, runtime)
    except SparkError as exc:
        warning(f"could not read the model catalog yet ({exc.code}); `spark run` will retry")
        return
    console.print(f"  models on this plan ({len(models)}):")
    for m in models[:20]:
        console.print(f"    • {m}", markup=False, highlight=False)
    hint(
        f"set remote_model in ~/.local/share/spark/models/{runtime}.toml",
        "the model listed first to consumers",
    )
    hint(f"spark run {runtime}", "serve it")


@click.command(name="logout")
@click.argument("runtime")
def logout_command(runtime: str):
    """Revoke RUNTIME's ChatGPT session and clear its tokens."""
    from ..secrets.oauth import logout, oauth_lock_path

    ctx = build_context()
    spec = _oauth_runtime(ctx, runtime).server.provider_oauth
    if logout(ctx.secrets, spec, runtime, oauth_lock_path(ctx.paths, spec)):
        success(f"revoked and cleared the ChatGPT session for {runtime}")
    else:
        console.print(f"[dim]no active session for {runtime}; tokens cleared[/dim]")
