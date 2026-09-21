"""Disk-first model inventory: what can this host actually serve?

The registry (``models/*.toml``) is a list of *intentions* — one file per model
someone once downloaded or hand-wrote. It outlives its weights silently: reclaim
a snapshot and the entry keeps advertising a model that cannot launch, while a
model sitting complete in the HF hub cache (or the spark store) stays invisible
because nobody registered it.

This module is the missing ground truth. It scans the two weight locations on
disk — the spark store and the Hugging Face hub cache — classifies what it finds
into *servable* (a generative model a runtime can serve) vs *cached but not
servable* (embedders, speech-to-text, TTS, …), and joins that against the
registry:

* ``runnable``        — servable weights on disk, registered or not. This is the
  only list the launch path, the default ``spark list``, and completion offer.
* ``missing``         — registry entries asserting weights that are absent. A
  defect to fix (re-fetch) or drop (``spark forget``), never a launch target.
* ``external``        — runtimes that own their weights (routers, daemons).
* ``non_servable``    — cached repos spark cannot serve (shown as a one-line
  count, never offered).

``spark run`` resolves against ``runnable`` first, so an unregistered-but-present
model launches via a synthesized ephemeral entry — no registration dance. ``spark
adopt`` promotes such a model into the registry when the operator wants it kept.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config.paths import SparkPaths
from .config.schema import ModelEntry
from .hf_cache import bytes_in, hf_cache_root, snapshot_dir

#: Marker files/dirs that identify a snapshot as a sentence-transformers
#: embedding model rather than a generative one.
_EMBEDDING_MARKERS = {"modules.json", "1_Pooling", "sentence_bert_config.json"}

#: model_type values that are never chat-servable.
_STT_TYPES = {"parakeet", "whisper", "wav2vec2", "hubert", "conformer"}
_EMBEDDING_TYPES = {"bert", "xlm-roberta", "roberta", "distilbert", "albert", "electra"}

#: model_type for PrismML Hadamard packs. Generative MLX weights, but only the
#: prism_ml backend (vendor loader + shim) can serve them — the format tag is
#: what routes them there instead of to stock mlx_lm (which answers
#: "Model type prism_hadamard_qwen35 not supported").
_PRISM_TYPE = "prism_hadamard_qwen35"

_QUANT_RE = re.compile(r"\b(q[2-8])(?:_[a-z0-9]+)?\b", re.I)


@dataclass(frozen=True)
class DiscoveredModel:
    """Servable-or-cached weights found on disk, registry or not."""

    #: What to show and accept on the CLI. Hub models use the full
    #: ``org/name`` repo id (it is directly runnable); store models use the
    #: directory name.
    display_id: str
    #: "hub" | "store".
    source: str
    #: Snapshot dir (hub) or store dir, as a string.
    location: str
    #: HF repo id when the weights live in the hub cache, else "".
    repo: str = ""
    bytes_present: int = 0
    #: "llm" | "embedding" | "stt" | "tts" | "unknown" | "partial".
    kind: str = "unknown"
    #: Human reason when not servable ("" when servable).
    reason: str = ""
    #: Format guess for backend selection ("mlx" | "gguf" | "any").
    model_format: str = "any"
    quant: str = ""
    #: Registry id when these weights back a registered entry, else "".
    registry_id: str = ""

    @property
    def servable(self) -> bool:
        return self.kind == "llm"

    @property
    def registered(self) -> bool:
        return bool(self.registry_id)


def _read_config_type(snap: Path) -> tuple[str, list[str]]:
    """(model_type, architectures) from a snapshot's config.json, tolerant."""
    try:
        data = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", []
    if not isinstance(data, dict):
        return "", []
    mt = data.get("model_type") or ""
    archs = data.get("architectures") or []
    return str(mt), [str(a) for a in archs] if isinstance(archs, list) else []


def classify_snapshot(snap: Path) -> tuple[str, str]:
    """Classify a hub-cache snapshot: (kind, reason).

    ``llm`` (reason "") means a generative model spark can serve. Anything else
    is cached-but-not-servable and must never be offered as a launch target.
    Never raises: an unreadable snapshot is ``unknown``, not an error.
    """
    try:
        names = {p.name for p in snap.iterdir()}
    except OSError:
        return "unknown", "snapshot is not readable"
    if any(n.lower().endswith(".gguf") for n in names):
        return "llm", ""
    if _EMBEDDING_MARKERS & names:
        return "embedding", "sentence-transformers embedding model, not a chat model"
    if "speech_tokenizer" in names:
        return "tts", "speech-synthesis model, not a chat model"
    model_type, archs = _read_config_type(snap)
    mt = model_type.lower()
    if mt == _PRISM_TYPE:
        return "llm", ""
    if "tts" in mt:
        return "tts", "speech-synthesis model, not a chat model"
    if mt in _STT_TYPES:
        return "stt", "speech-recognition model, not a chat model"
    if mt in _EMBEDDING_TYPES or any(
        a in ("BertModel", "XLMRobertaModel", "RobertaModel") for a in archs
    ):
        return "embedding", "embedding/classifier model, not a chat model"
    if any(a.endswith(("ForSequenceClassification", "ForMaskedLM")) for a in archs):
        return "embedding", "embedding/classifier model, not a chat model"
    if any(a.endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs):
        return "llm", ""
    if "tokenizer.model" in names and "tokenizer.json" not in names and not mt:
        return "stt", "speech-recognition model, not a chat model"
    weights = any(
        n == "model.safetensors"
        or n.endswith(".safetensors")
        or n.endswith(".bin")
        for n in names
    )
    if weights and (snap / "config.json").is_file():
        # Generative by elimination: embedding/speech families are excluded
        # above, so weights + config + tokenizer is an LLM until proven otherwise.
        if "tokenizer.json" in names or "tokenizer_config.json" in names:
            return "llm", ""
        return "unknown", "weights present but no tokenizer found"
    return "unknown", "no recognizable weights"


def _guess_format(repo: str, snap: Path) -> str:
    if repo and "gguf" in repo.lower():
        return "gguf"
    try:
        if any(p.suffix == ".gguf" for p in snap.iterdir()):
            return "gguf"
    except OSError:
        pass
    mt, _ = _read_config_type(snap)
    if mt.lower() == _PRISM_TYPE:
        return "prism"
    if repo and "mlx" in repo.lower():
        return "mlx"
    return "any"


def _guess_quant(repo: str) -> str:
    m = _QUANT_RE.search(repo or "")
    return m.group(1).lower() if m else ""


def scan_hub_cache(*, with_bytes: bool = True) -> list[DiscoveredModel]:
    """Every repo directory in the HF hub cache with a resolvable snapshot.

    Snapshots without a ``config.json`` (reclaimed or mid-download) are skipped:
    nothing servable can be said about them. Never raises — an unreadable cache
    root yields an empty list.
    """
    try:
        root = hf_cache_root()
        if not root.is_dir():
            return []
        dirs = sorted(d for d in root.iterdir() if d.name.startswith("models--"))
    except OSError:
        return []
    out: list[DiscoveredModel] = []
    for d in dirs:
        repo = d.name[len("models--"):].replace("--", "/")
        try:
            snap = snapshot_dir(repo)
        except OSError:
            continue
        if snap is None:
            continue
        kind, reason = classify_snapshot(snap)
        out.append(
            DiscoveredModel(
                display_id=repo,
                repo=repo,
                source="hub",
                location=str(snap),
                bytes_present=bytes_in(snap) if with_bytes else 0,
                kind=kind,
                reason=reason,
                model_format=_guess_format(repo, snap),
                quant=_guess_quant(repo),
            )
        )
    return out


def scan_store(paths: SparkPaths, *, with_bytes: bool = True) -> list[DiscoveredModel]:
    """Every model directory in the spark store.

    A directory with surviving ``*.incomplete`` resume fragments is ``partial``:
    present but not runnable until the download is resumed.
    """
    from .registry.store import dir_bytes  # local import: registry must not import inventory

    store = paths.model_store_dir
    try:
        if not store.is_dir():
            return []
        dirs = sorted(d for d in store.iterdir() if d.is_dir())
    except OSError:
        return []
    out: list[DiscoveredModel] = []
    for d in dirs:
        dl_dir = d / ".cache" / "huggingface" / "download"
        try:
            incomplete = list(dl_dir.glob("*.incomplete")) if dl_dir.is_dir() else []
        except OSError:
            incomplete = []
        if incomplete:
            out.append(
                DiscoveredModel(
                    display_id=d.name,
                    source="store",
                    location=str(d),
                    bytes_present=dir_bytes(d) if with_bytes else 0,
                    kind="partial",
                    reason=f"incomplete download ({len(incomplete)} resume fragment(s))",
                )
            )
            continue
        out.append(
            DiscoveredModel(
                display_id=d.name,
                source="store",
                location=str(d),
                bytes_present=dir_bytes(d) if with_bytes else 0,
                kind="llm",
                model_format="any",
                quant=_guess_quant(d.name),
            )
        )
    return out


@dataclass
class Inventory:
    """The join of disk truth and registry intentions, categorized for the CLI."""

    #: (entry, availability-state) for registered models whose weights are present.
    runnable_registered: list[tuple[ModelEntry, object]] = field(default_factory=list)
    #: Servable weights on disk with no registry entry. Runnable right now.
    runnable_discovered: list[DiscoveredModel] = field(default_factory=list)
    #: (entry, availability) for entries asserting absent weights.
    missing: list[tuple[ModelEntry, object]] = field(default_factory=list)
    #: (entry, availability) for runtimes that own their weights.
    external: list[tuple[ModelEntry, object]] = field(default_factory=list)
    #: (entry, availability) for entries asserting nothing.
    unverifiable: list[tuple[ModelEntry, object]] = field(default_factory=list)
    #: Cached repos spark cannot serve. Never offered; reported as a count.
    non_servable: list[DiscoveredModel] = field(default_factory=list)

    @property
    def runnable_ids(self) -> list[str]:
        ids = [e.id for e, _ in self.runnable_registered]
        ids += [m.display_id for m in self.runnable_discovered]
        ids += [e.id for e, _ in self.external]
        return ids


def build_inventory(
    paths: SparkPaths, config=None, *, with_bytes: bool = True
) -> Inventory:
    """Scan disk, join against the registry. Never raises on I/O trouble."""
    from .registry import list_models
    from .registry.availability import (
        EXTERNAL,
        MISSING,
        UNVERIFIABLE,
        availability_map,
    )

    inv = Inventory()
    try:
        entries = list_models(paths)
    except OSError:
        entries = []
    avail = availability_map(entries, paths, config=config)

    by_hf_repo: dict[str, ModelEntry] = {}
    by_path: dict[str, ModelEntry] = {}
    for e in entries:
        if e.hf_repo:
            by_hf_repo[e.hf_repo.lower()] = e
        if e.path:
            try:
                by_path[str(Path(e.path).expanduser())] = e
            except (OSError, RuntimeError):
                continue

    for e in entries:
        a = avail.get(e.id)
        if a is None:
            continue
        if a.state == MISSING:
            inv.missing.append((e, a))
        elif a.state == EXTERNAL:
            inv.external.append((e, a))
        elif a.state == UNVERIFIABLE:
            inv.unverifiable.append((e, a))
        else:
            inv.runnable_registered.append((e, a))

    claimed_locations = set(by_path)
    for disc in scan_hub_cache(with_bytes=with_bytes):
        owner = by_hf_repo.get(disc.repo.lower())
        if owner is not None:
            # Already covered by the registered row; skip unless it is servable
            # but the entry is missing (cannot happen: availability resolves the
            # same snapshot) — keep the join exact, never double-list.
            continue
        if disc.servable:
            inv.runnable_discovered.append(disc)
        else:
            inv.non_servable.append(disc)
    for disc in scan_store(paths, with_bytes=with_bytes):
        if disc.location in claimed_locations:
            continue
        if disc.servable:
            inv.runnable_discovered.append(
                DiscoveredModel(
                    display_id=disc.display_id,
                    repo=disc.repo,
                    source=disc.source,
                    location=disc.location,
                    bytes_present=disc.bytes_present,
                    kind=disc.kind,
                    reason=disc.reason,
                    model_format=disc.model_format,
                    quant=disc.quant,
                )
            )
        elif disc.kind == "partial":
            # A partial store dir is neither runnable nor cached-elsewhere; the
            # store-issues warning owns it. Keep it out of every list here.
            continue
        else:
            inv.non_servable.append(disc)
    inv.runnable_discovered.sort(key=lambda m: m.display_id.lower())
    inv.non_servable.sort(key=lambda m: m.display_id.lower())
    return inv


def synthesize_entry(disc: DiscoveredModel) -> ModelEntry:
    """An ephemeral registry entry for unregistered-but-present weights.

    Never written to disk. Carries just enough for backend selection and the
    launch preflight: hub models resolve via ``hf_repo`` (the runtime reads the
    hub cache directly, no re-download), store models via ``path``.
    """
    if disc.source == "hub":
        return ModelEntry(
            id=disc.display_id,
            hf_repo=disc.repo,
            path="",
            model_format=disc.model_format,  # type: ignore[arg-type]
            quant=disc.quant,
            research_status="pending",
            notes="ephemeral: weights found on disk, not in the registry "
            "(persist with `spark adopt`)",
        )
    return ModelEntry(
        id=disc.display_id,
        hf_repo="",
        path=disc.location,
        model_format="any",
        quant=disc.quant,
        research_status="pending",
        notes="ephemeral: weights found in the spark store, not in the registry "
        "(persist with `spark adopt`)",
    )


@dataclass(frozen=True)
class RunResolution:
    entry: ModelEntry
    ephemeral: bool
    via: str  # "registry" | "hub" | "store"


def resolve_for_run(query: str, paths: SparkPaths, config=None) -> RunResolution | None:
    """Resolve a run target against disk truth first, registry second.

    Order: registry exact id/alias → hub repo exact → store dirname exact →
    unique prefix/substring across all runnable names. Returns None when nothing
    matches (the caller raises with suggestions from the inventory).
    """
    q = (query or "").strip()
    if not q:
        return None
    ql = q.lower()
    inv = build_inventory(paths, config=config, with_bytes=False)
    reg_by_id = {e.id.lower(): e for e, _ in inv.runnable_registered}
    reg_by_id.update({e.id.lower(): e for e, _ in inv.external})
    reg_by_id.update({e.id.lower(): e for e, _ in inv.unverifiable})
    if ql in reg_by_id:
        return RunResolution(entry=reg_by_id[ql], ephemeral=False, via="registry")
    # Aliases of runnable/external entries.
    for e, _ in (*inv.runnable_registered, *inv.external, *inv.unverifiable):
        if any(a.lower() == ql for a in e.aliases):
            return RunResolution(entry=e, ephemeral=False, via="registry")

    hub = {m.display_id.lower(): m for m in inv.runnable_discovered if m.source == "hub"}
    if ql in hub:
        return RunResolution(entry=synthesize_entry(hub[ql]), ephemeral=True, via="hub")
    store = {
        m.display_id.lower(): m for m in inv.runnable_discovered if m.source == "store"
    }
    if ql in store:
        return RunResolution(entry=synthesize_entry(store[ql]), ephemeral=True, via="store")

    # Unique prefix, then unique substring, across every runnable name.
    names: list[tuple[str, RunResolution]] = []
    for e, _ in (*inv.runnable_registered, *inv.external, *inv.unverifiable):
        names.append(
            (e.id.lower(), RunResolution(entry=e, ephemeral=False, via="registry"))
        )
    for m in inv.runnable_discovered:
        names.append(
            (
                m.display_id.lower(),
                RunResolution(entry=synthesize_entry(m), ephemeral=True, via=m.source),
            )
        )
    prefix = [(n, r) for n, r in names if n.startswith(ql)]
    if len(prefix) == 1:
        return prefix[0][1]
    substr = [(n, r) for n, r in names if ql in n]
    if len(substr) == 1:
        return substr[0][1]
    # Also accept the short tail of a hub repo ("bonsai-8b" for
    # "prism-ml/Ternary-Bonsai-8B-mlx-2bit") when it is unique.
    tails = [
        (m.display_id.lower().split("/")[-1], m)
        for m in inv.runnable_discovered
        if m.source == "hub"
    ]
    tail_hit = [m for t, m in tails if t == ql or t.startswith(ql)]
    if len(tail_hit) == 1:
        return RunResolution(
            entry=synthesize_entry(tail_hit[0]), ephemeral=True, via="hub"
        )
    return None


__all__ = [
    "DiscoveredModel",
    "Inventory",
    "RunResolution",
    "build_inventory",
    "classify_snapshot",
    "resolve_for_run",
    "scan_hub_cache",
    "scan_store",
    "synthesize_entry",
]
