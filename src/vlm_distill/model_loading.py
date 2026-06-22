from __future__ import annotations

import importlib.util
from pathlib import Path


LOCAL_MODEL_CACHE_ROOT = Path("~/vlm_distill/models").expanduser()


_ATTN_ALIASES = {
    "sdpa": "sdpa",
    "eager": "eager",
    "fa2": "flash_attention_2",
    "flash": "flash_attention_2",
    "flash2": "flash_attention_2",
    "flash_attn_2": "flash_attention_2",
    "flash-attn-2": "flash_attention_2",
    "flash_attention_2": "flash_attention_2",
}


def resolve_attn_implementation(value: str | None) -> str:
    raw = (value or "sdpa").strip().lower()
    if raw not in _ATTN_ALIASES:
        raise ValueError(
            f"Unsupported attn_implementation={value!r}. "
            "Use one of: sdpa, eager, flash_attention_2."
        )

    resolved = _ATTN_ALIASES[raw]
    if resolved == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError(
            "attn_implementation='flash_attention_2' requires the `flash_attn` package "
            "to be installed in the active environment."
        )
    return resolved


def apply_attn_implementation(model_kwargs: dict, value: str | None) -> dict:
    model_kwargs["attn_implementation"] = resolve_attn_implementation(value)
    return model_kwargs


def resolve_model_path(
    model_name_or_path: str,
    *,
    cache_root: Path = LOCAL_MODEL_CACHE_ROOT,
) -> str:
    """Resolve a config model value to a local loadable path."""
    raw = str(model_name_or_path)
    candidate = Path(raw).expanduser()

    if candidate.exists():
        return str(_resolve_loadable_snapshot(candidate))

    if _looks_like_hf_repo_id(raw):
        cache_dir = cache_root / f"models--{raw.replace('/', '--')}"
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Local model cache directory does not exist for {raw!r}: {cache_dir}. "
                f"Expected Hugging Face cache-style directory under {cache_root}."
            )

        return str(_resolve_loadable_snapshot(cache_dir))

    if _looks_like_local_path(raw):
        raise FileNotFoundError(
            f"Configured model path does not exist: {candidate}. "
            "Set the config to an existing local model directory."
        )

    raise FileNotFoundError(
        f"Could not resolve model {raw!r}. Use a local path or a Hugging Face repo id "
        f"with a matching cache directory under {cache_root}."
    )


def _looks_like_local_path(value: str) -> bool:
    return (
        value.startswith(("~", ".", "/"))
        or "\\" in value
        or "/" in value
        or (len(value) >= 2 and value[1] == ":")
    )


def _looks_like_hf_repo_id(value: str) -> bool:
    parts = value.split("/")
    if len(parts) != 2:
        return False
    return all(part and not part.startswith(".") and "\\" not in part for part in parts)


def _resolve_loadable_snapshot(path: Path) -> Path:
    if (path / "config.json").exists():
        return path

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"Model directory is not directly loadable and has no snapshots directory: {path}. "
            "Expected config.json in the model directory or snapshots/<commit>/config.json."
        )

    refs_main = path / "refs" / "main"
    if refs_main.exists():
        commit = refs_main.read_text(encoding="utf-8").strip()
        snapshot_path = snapshots_dir / commit
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Snapshot referenced by {refs_main} does not exist: {snapshot_path}."
            )
        if not (snapshot_path / "config.json").exists():
            raise FileNotFoundError(
                f"Snapshot path is not loadable because config.json is missing: {snapshot_path}."
            )
        return snapshot_path

    snapshots = sorted(
        snapshot for snapshot in snapshots_dir.iterdir()
        if snapshot.is_dir() and (snapshot / "config.json").exists()
    )
    if len(snapshots) == 1:
        return snapshots[0]
    if not snapshots:
        raise FileNotFoundError(
            f"No loadable snapshots found under {snapshots_dir}. "
            "Expected at least one snapshots/<commit>/config.json."
        )
    raise FileNotFoundError(
        f"Multiple loadable snapshots found under {snapshots_dir} and no refs/main was found. "
        "Add refs/main or configure an explicit snapshots/<commit> path."
    )
