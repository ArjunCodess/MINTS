"""Model loading and TransformerLens wrapping for nucleotide transformers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_CONFIG, ModelConfig
from .utils import set_reproducibility_seed


@dataclass
class LoadedModelBundle:
    """Container for Hugging Face and TransformerLens model components."""

    tokenizer: Any
    hf_model: Any
    hooked_model: Any
    model_name: str
    device: str


class ModelCompatibilityError(RuntimeError):
    """Raised when TransformerLens cannot wrap the selected HF model."""


def _resolve_device(device: str) -> str:
    """Resolve `auto` to the best locally available torch device."""

    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hf_components(config: ModelConfig = DEFAULT_CONFIG.model) -> tuple[Any, Any, str]:
    """Load the Hugging Face tokenizer and PyTorch model."""

    from transformers import AutoModel, AutoTokenizer

    device = _resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
    )
    hf_model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
    )
    hf_model.eval()
    hf_model.to(device)
    return tokenizer, hf_model, device


def _select_hooked_encoder_class() -> Any:
    """Import TransformerLens' encoder wrapper."""

    try:
        from transformer_lens import HookedEncoder
    except ImportError as exc:
        raise ImportError(
            "TransformerLens is not installed. Install `transformer-lens` from "
            "requirements.txt before running model wrapping."
        ) from exc
    return HookedEncoder


def _call_from_pretrained(hooked_encoder_cls: Any, kwargs: dict[str, Any]) -> Any:
    """Call HookedEncoder.from_pretrained with supported keyword arguments."""

    signature = inspect.signature(hooked_encoder_cls.from_pretrained)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        return hooked_encoder_cls.from_pretrained(**kwargs)

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return hooked_encoder_cls.from_pretrained(**filtered)


def load_hooked_encoder(config: ModelConfig = DEFAULT_CONFIG.model) -> LoadedModelBundle:
    """Load DNABERT-2 and wrap it with TransformerLens `HookedEncoder`.

    If the installed TransformerLens version cannot consume DNABERT-2's custom
    Hugging Face code, this function raises a clear compatibility error rather
    than returning a partially instrumented model.
    """

    set_reproducibility_seed(DEFAULT_CONFIG.data.seed)
    tokenizer, hf_model, device = load_hf_components(config)
    hooked_encoder_cls = _select_hooked_encoder_class()
    kwargs = {
        "model_name": config.model_name,
        "hf_model": hf_model,
        "tokenizer": tokenizer,
        "device": device,
        "trust_remote_code": config.trust_remote_code,
        "revision": config.revision,
    }
    try:
        hooked_model = _call_from_pretrained(hooked_encoder_cls, kwargs)
    except Exception as exc:
        raise ModelCompatibilityError(
            "TransformerLens could not wrap DNABERT-2 with HookedEncoder in the "
            "current environment. The pipeline fails fast here so downstream analysis does not "
            "run on an uninstrumented model. Upgrade TransformerLens or add the "
            "custom PyTorch-hook fallback for this HF model."
        ) from exc

    hooked_model.eval()
    return LoadedModelBundle(
        tokenizer=tokenizer,
        hf_model=hf_model,
        hooked_model=hooked_model,
        model_name=config.model_name,
        device=device,
    )


def summarize_hook_points(hooked_model: Any, limit: int = 50) -> list[str]:
    """Return available hook point names for compatibility manifests."""

    hook_dict = getattr(hooked_model, "hook_dict", None)
    if not hook_dict:
        return []
    return list(hook_dict.keys())[:limit]
