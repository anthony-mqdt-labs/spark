"""Typed error factories for `spark adopt` misses.

Adopt classification is registry-domain knowledge (registry rows, inventory
state, hub/store caches), not CLI presentation: the click layer only reports the
error these factories build. Keeping it here lets `cli/app.py` stay thin and
gives other entry points the same state-aware diagnostics.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import AlreadyRegisteredError, AmbiguousModelError, ModelNotFoundError, SparkError
from ..hf_cache import hf_cache_root, repo_dir, snapshot_dir
from .models import fuzzy_match
from .store import scan_store


def adopt_registry_exact(target: str, registered):
    """The registry row this query already names (id, alias, repo or path)."""
    tl = target.lower()
    for entry, avail in registered:
        if entry.id.lower() == tl:
            return entry, avail
        if any(alias.lower() == tl for alias in entry.aliases):
            return entry, avail
        if (entry.hf_repo or "").lower() == tl:
            return entry, avail
        loc = getattr(avail, "location", "") or ""
        if loc and (loc == target or loc.lower() == tl):
            return entry, avail
        if entry.path and (entry.path == target or entry.path.lower() == tl):
            return entry, avail
    return None


def adopt_non_servable_hits(target: str, inv):
    tl = target.lower()
    exact = [m for m in inv.non_servable if m.display_id.lower() == tl]
    if exact:
        return exact
    tails = [m for m in inv.non_servable if m.display_id.lower().split("/")[-1] == tl]
    if tails:
        return tails
    prefix = [m for m in inv.non_servable if m.display_id.lower().startswith(tl)]
    return prefix or [m for m in inv.non_servable if tl in m.display_id.lower()]


def adopt_registered_error(repo: str, entry, avail):
    """The exact query is already registered — explain its state precisely."""
    backend = entry.backend or "auto"
    research = entry.research_status
    state = getattr(avail, "state", "?")
    detail = getattr(avail, "detail", "")
    location = getattr(avail, "location", "") or ""
    if state == "missing":
        weights = f"weights missing ({detail})" if detail else "weights missing"
    elif state in ("local", "hub"):
        weights = f"weights present ({state}{f': {location}' if location else ''})"
    elif state == "external":
        weights = f"weights owned by '{entry.backend}' runtime"
    elif state == "unverifiable":
        weights = "entry declares neither path nor hf_repo"
    else:
        weights = f"weights state: {state}"
    message = (
        f"'{repo}' is already registered as '{entry.id}' — nothing to adopt.\n"
        f"backend: {backend} · research: {research} · {weights}"
    )
    if state == "missing":
        remediation = [f"Fetch weights: spark download {entry.hf_repo}"] if entry.hf_repo else []
        remediation += [f"Drop the entry: spark forget {entry.id}"]
    elif research == "pending":
        # `pending` is research state, not download state — the original confusion.
        remediation = [
            f"Weights are on disk — no adopt needed. Research next: spark research {entry.id}",
            f"Or just run it: spark run {entry.id}",
        ]
    else:
        remediation = [f"Run it: spark run {entry.id}"]
    return AlreadyRegisteredError(
        message,
        remediation=remediation,
        context={
            "query": repo,
            "model_id": entry.id,
            "backend": entry.backend,
            "research_status": research,
            "availability": state,
            "location": location,
        },
    )


def adopt_miss_error(repo: str, inv, ctx) -> SparkError:
    """State-aware error for `spark adopt <query>` when nothing was adopted.

    The old message always said "nothing adoptable", even when the query was
    already registered (the common confusion: the `status` column shows
    research state — `pending` — not download state). Report what the query
    actually is: already registered (with backend/research/weights state),
    cached-but-not-servable, a partial store download, an incomplete hub
    snapshot, or genuinely unknown.
    """
    target = repo.strip()
    tl = target.lower()

    registered = [
        (e, a)
        for group in (
            inv.runnable_registered,
            inv.missing,
            inv.external,
            inv.unverifiable,
        )
        for e, a in group
    ]

    hit = adopt_registry_exact(target, registered)
    if hit is not None:
        return adopt_registered_error(repo, *hit)

    # Ambiguous registry query (adopt takes one model).
    _hit, candidates = fuzzy_match(repo, [(e.id, e) for e, _ in registered])
    if len(candidates) > 1:
        ids = sorted({e.id for e in candidates})
        return AmbiguousModelError(
            f"'{repo}' matches multiple registered models.",
            remediation=[f"Be more specific: {', '.join(ids)}"],
            context={"query": repo, "matches": ids},
        )

    # Cached repos spark cannot serve (embedders, speech). Never adoptable,
    # but the query *is* on disk — say so instead of "nothing matches".
    ns_hits = adopt_non_servable_hits(target, inv)
    if ns_hits:
        m = ns_hits[0]
        others = f" (+{len(ns_hits) - 1} more)" if len(ns_hits) > 1 else ""
        return ModelNotFoundError(
            f"'{repo}' is cached as '{m.display_id}'{others} but spark cannot serve it "
            f"({m.kind}) — {m.reason}. Nothing to adopt.",
            remediation=[
                "Adoptable models are the unregistered runnable rows of: spark list",
                "Inspect the hidden cache: spark list --all",
            ],
            context={"query": repo, "kind": m.kind, "reason": m.reason},
        )

    # Partial / orphaned store dirs own their message (resume, don't adopt).
    try:
        issues = scan_store(ctx.paths)
    except OSError:
        issues = []
    for issue in issues:
        if (
            issue.model_id.lower() == tl
            or str(issue.path) == target
            or issue.model_id.lower().split("/")[-1] == tl
        ):
            if issue.incomplete_files:
                resume = (
                    f"spark download {issue.hf_repo}"
                    if issue.hf_repo
                    else f"spark download <hf-repo> --id {issue.model_id}"
                )
                return ModelNotFoundError(
                    f"'{repo}' is an incomplete download in the model store "
                    f"({issue.incomplete_files} resume fragment(s)) — resume it, don't adopt it.",
                    remediation=[
                        f"Resume: {resume}",
                        f"Clean: rm -rf {issue.path}",
                    ],
                    context={"query": repo, "path": str(issue.path)},
                )
            return ModelNotFoundError(
                f"'{repo}' is in the model store but has no servable snapshot "
                f"({issue.path}) — nothing to adopt yet.",
                remediation=[
                    "Adoptable models are the unregistered runnable rows of: spark list",
                    f"Clean: rm -rf {issue.path}",
                ],
                context={"query": repo, "path": str(issue.path)},
            )

    # Hub cache dir exists but no complete snapshot (mid-download / reclaimed).
    if "/" in target:
        try:
            if repo_dir(target).is_dir() and snapshot_dir(target) is None:
                return ModelNotFoundError(
                    f"'{repo}' is in the HF hub cache but has no complete snapshot "
                    f"(no config.json) — interrupted download or reclaimed weights.",
                    remediation=[
                        f"Resume: spark download {target}",
                        "Adoptable models are the unregistered runnable rows of: spark list",
                    ],
                    context={"query": repo},
                )
        except OSError:
            pass
        # Case-insensitive hub hit (hub ids are case-sensitive on disk, but the
        # operator often retypes them lowercased): point at the real row.
        try:
            root = hf_cache_root()
            want = target.replace("/", "--").lower()
            if root.is_dir():
                for d in root.iterdir():
                    if d.name.startswith("models--") and d.name[len("models--") :].lower() == want:
                        real = d.name[len("models--") :].replace("--", "/")
                        return ModelNotFoundError(
                            f"Nothing adoptable matches '{repo}' (case differs from cached '{real}').",
                            remediation=[
                                f"Check the exact row: spark list | grep -i {Path(target).name}",
                                "Adoptable models are the unregistered runnable rows of: spark list",
                            ],
                            context={"query": repo, "cached": real},
                        )
        except OSError:
            pass

    if inv.runnable_discovered:
        sample = ", ".join(m.display_id for m in inv.runnable_discovered[:5])
        more = (
            f" (+{len(inv.runnable_discovered) - 5} more)"
            if len(inv.runnable_discovered) > 5
            else ""
        )
        return ModelNotFoundError(
            f"Nothing adoptable matches '{repo}'.",
            remediation=[
                f"Adoptable now: {sample}{more}",
                "Adoptable models are the unregistered rows of: spark list",
                f"Or download it first: spark download {repo}",
            ],
            context={"query": repo},
        )
    return ModelNotFoundError(
        f"Nothing adoptable matches '{repo}'.",
        remediation=[
            "Adoptable models are the unregistered rows of: spark list",
            f"Or download it first: spark download {repo}",
        ],
        context={"query": repo},
    )
