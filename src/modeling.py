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
    instrumentation_backend: str
    instrumentation_error: str | None = None


class HuggingFaceHookAdapter:
    """Minimal hook-based adapter for HF models unsupported by TransformerLens."""

    def __init__(self, model: Any):
        self.model = model
        self.hook_dict = {
            name: module
            for name, module in model.named_modules()
            if name and any(part in name.lower() for part in ("encoder", "attention", "layer", "embedding"))
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def eval(self) -> "HuggingFaceHookAdapter":
        self.model.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "HuggingFaceHookAdapter":
        self.model.to(*args, **kwargs)
        return self

    def run_with_cache(self, *model_args: Any, names_filter: Any = None, **model_kwargs: Any):
        """Run the HF model while caching selected module outputs."""

        cache: dict[str, Any] = {}
        handles = []

        def include_name(name: str) -> bool:
            if names_filter is None:
                return True
            if isinstance(names_filter, str):
                return names_filter == name
            if callable(names_filter):
                return bool(names_filter(name))
            return name in names_filter

        def make_hook(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                if hasattr(value, "detach"):
                    value = value.detach()
                cache[name] = value

            return hook

        for name, module in self.hook_dict.items():
            if include_name(name):
                handles.append(module.register_forward_hook(make_hook(name)))
        try:
            output = self.model(*model_args, **model_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        return output, cache


class ModelCompatibilityError(RuntimeError):
    """Raised when TransformerLens cannot wrap the selected HF model."""


def _resolve_device(device: str) -> str:
    """Resolve `auto` to the best locally available torch device."""

    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _patch_dnabert_alibi_builder(model_name: str, hf_config: Any) -> None:
    """Patch DNABERT-2 ALiBi construction for Transformers meta-device loading.

    Recent Transformers versions may instantiate remote-code models under a
    meta-device context. DNABERT-2's `BertEncoder.rebuild_alibi_tensor` leaves
    one intermediate tensor on CPU when `device=None`, which causes a CPU/meta
    device mismatch during model construction. This patch preserves the original
    method but defaults the device to the existing `self.alibi.device`.
    """

    auto_map = getattr(hf_config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModel")
    if class_ref != "bert_layers.BertModel":
        return

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return

    model_class = get_class_from_dynamic_module(class_ref, model_name)
    module = __import__(model_class.__module__, fromlist=["BertEncoder"])
    encoder_class = getattr(module, "BertEncoder", None)
    if encoder_class is None or getattr(encoder_class, "_mints_alibi_patch", False):
        return

    original = encoder_class.rebuild_alibi_tensor

    def patched_rebuild_alibi_tensor(self, size: int, device: Any = None):
        if device is None:
            device = getattr(getattr(self, "alibi", None), "device", None)
        return original(self, size=size, device=device)

    encoder_class.rebuild_alibi_tensor = patched_rebuild_alibi_tensor
    encoder_class._mints_alibi_patch = True


def _disable_dnabert_triton_attention(model_name: str, hf_config: Any) -> None:
    """Force DNABERT-2 to use its PyTorch attention fallback.

    DNABERT-2 ships a Triton FlashAttention implementation whose `tl.dot`
    signature is incompatible with newer Triton builds on this environment.
    The remote attention module already has a maintained PyTorch fallback when
    `flash_attn_qkvpacked_func` is `None`; this function selects that path.
    """

    auto_map = getattr(hf_config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModel")
    if class_ref != "bert_layers.BertModel":
        return

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return

    model_class = get_class_from_dynamic_module(class_ref, model_name)
    module = __import__(model_class.__module__, fromlist=["flash_attn_qkvpacked_func"])
    if hasattr(module, "flash_attn_qkvpacked_func"):
        module.flash_attn_qkvpacked_func = None


def _rebuild_alibi_on_device(hf_model: Any, device: str) -> None:
    """Rebuild DNABERT-2's ALiBi buffer on the actual runtime device."""

    encoder = getattr(hf_model, "encoder", None)
    rebuild = getattr(encoder, "rebuild_alibi_tensor", None)
    if rebuild is None:
        return
    size = getattr(getattr(hf_model, "config", None), "alibi_starting_size", None)
    if size is None:
        return
    rebuild(size=size, device=device)


def load_hf_components(config: ModelConfig = DEFAULT_CONFIG.model) -> tuple[Any, Any, str]:
    """Load the Hugging Face tokenizer and PyTorch model."""

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    device = _resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
    )
    hf_config = AutoConfig.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
    )
    if not hasattr(hf_config, "pad_token_id") or hf_config.pad_token_id is None:
        hf_config.pad_token_id = tokenizer.pad_token_id
    if not hasattr(hf_config, "bos_token_id") and tokenizer.cls_token_id is not None:
        hf_config.bos_token_id = tokenizer.cls_token_id
    if not hasattr(hf_config, "eos_token_id") and tokenizer.sep_token_id is not None:
        hf_config.eos_token_id = tokenizer.sep_token_id

    _patch_dnabert_alibi_builder(config.model_name, hf_config)
    _disable_dnabert_triton_attention(config.model_name, hf_config)
    hf_model = AutoModel.from_pretrained(
        config.model_name,
        config=hf_config,
        trust_remote_code=config.trust_remote_code,
        revision=config.revision,
    )
    _rebuild_alibi_on_device(hf_model, device)
    if hasattr(hf_model, "pooler"):
        hf_model.pooler = None
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
        adapter = HuggingFaceHookAdapter(hf_model)
        return LoadedModelBundle(
            tokenizer=tokenizer,
            hf_model=hf_model,
            hooked_model=adapter,
            model_name=config.model_name,
            device=device,
            instrumentation_backend="huggingface_forward_hooks",
            instrumentation_error=f"{type(exc).__name__}: {exc}",
        )

    hooked_model.eval()
    return LoadedModelBundle(
        tokenizer=tokenizer,
        hf_model=hf_model,
        hooked_model=hooked_model,
        model_name=config.model_name,
        device=device,
        instrumentation_backend="transformer_lens_hooked_encoder",
    )


def summarize_hook_points(hooked_model: Any, limit: int = 50) -> list[str]:
    """Return available hook point names for compatibility manifests."""

    hook_dict = getattr(hooked_model, "hook_dict", None)
    if not hook_dict:
        return []
    return list(hook_dict.keys())[:limit]
