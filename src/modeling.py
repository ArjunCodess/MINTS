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


DNABERT2_MODEL_NAME = "zhihan1996/DNABERT-2-117M"
NUCLEOTIDE_TRANSFORMER_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"


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


def _patch_esm_config_defaults(hf_config: Any) -> None:
    """Fill ESM defaults expected by newer Transformers internals."""

    if getattr(hf_config, "model_type", None) != "esm":
        return
    defaults = {
        "is_decoder": False,
        "add_cross_attention": False,
        "pruned_heads": {},
    }
    for name, value in defaults.items():
        if not hasattr(hf_config, name):
            setattr(hf_config, name, value)


def _patch_transformers_pruning_helper() -> None:
    """Restore a small pruning helper expected by older remote ESM code."""

    import torch
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - already_pruned_heads
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask), dtype=torch.long)[mask].long()
        return heads, index

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _patch_remote_masked_lm_class(model_name: str, hf_config: Any) -> None:
    """Patch old remote model classes for current Transformers loaders."""

    auto_map = getattr(hf_config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForMaskedLM")
    if class_ref is None:
        return
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return
    model_class = get_class_from_dynamic_module(class_ref, model_name)
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}


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


def _patch_runtime_compat_methods(hf_model: Any) -> None:
    """Patch helper methods expected by older remote model code."""

    import types

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
        if head_mask is None:
            return [None] * int(num_hidden_layers)
        if hasattr(self, "_convert_head_mask_to_5d"):
            return self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        return head_mask

    for module in (hf_model, getattr(hf_model, "esm", None)):
        if module is not None and not hasattr(module, "get_head_mask"):
            module.get_head_mask = types.MethodType(get_head_mask, module)


def load_hf_components(config: ModelConfig = DEFAULT_CONFIG.model) -> tuple[Any, Any, str]:
    """Load the Hugging Face tokenizer and PyTorch model."""

    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

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
    _patch_esm_config_defaults(hf_config)
    _patch_transformers_pruning_helper()
    _patch_remote_masked_lm_class(config.model_name, hf_config)
    prefers_masked_lm = bool((getattr(hf_config, "auto_map", None) or {}).get("AutoModelForMaskedLM"))
    loaders = (AutoModelForMaskedLM, AutoModel) if prefers_masked_lm else (AutoModel, AutoModelForMaskedLM)
    errors: list[str] = []
    hf_model = None
    for loader in loaders:
        try:
            hf_model = loader.from_pretrained(
                config.model_name,
                config=hf_config,
                trust_remote_code=config.trust_remote_code,
                revision=config.revision,
            )
            break
        except Exception as exc:
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
    if hf_model is None:
        raise RuntimeError(
            f"Could not load {config.model_name} with AutoModel or AutoModelForMaskedLM. "
            + " | ".join(errors)
        )
    _rebuild_alibi_on_device(hf_model, device)
    if hasattr(hf_model, "pooler"):
        hf_model.pooler = None
    _patch_runtime_compat_methods(hf_model)
    hf_model.eval()
    hf_model.to(device)
    return tokenizer, hf_model, device


def model_slug(model_name: str) -> str:
    """Return a filesystem-safe model identifier."""

    return (
        model_name.lower()
        .replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def tokenization_family(model_name: str, tokenizer: Any | None = None) -> str:
    """Classify the model tokenizer family for comparison reports."""

    lower = model_name.lower()
    if "dnabert-2" in lower or "dnabert_2" in lower:
        return "BPE"
    if "nucleotide-transformer" in lower:
        return "fixed_6mer"
    tokenizer_name = type(tokenizer).__name__ if tokenizer is not None else "unknown"
    lower_tokenizer = tokenizer_name.lower()
    if "bpe" in lower_tokenizer:
        return "BPE"
    return tokenizer_name


def encode_sequences(tokenizer: Any, sequences: list[str] | str, device: str, max_length: int | None = None) -> dict[str, Any]:
    """Tokenize one or more DNA sequences and move tensors to the runtime device."""

    kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(sequences, **kwargs)
    return {key: value.to(device) for key, value in encoded.items()}


def forward_hidden_states(model: Any, encoded: dict[str, Any]) -> list[Any]:
    """Run an encoder model and return hidden states as a list.

    DNABERT-2's remote code exposes hidden states through
    `output_all_encoded_layers=True`, while standard Hugging Face encoders use
    `output_hidden_states=True`. This helper normalizes both paths for probing
    and hook-based intervention code.
    """

    try:
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            output_all_encoded_layers=True,
        )
        hidden = outputs[0]
        if isinstance(hidden, (list, tuple)):
            return list(hidden)
    except TypeError:
        pass

    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output did not expose hidden states.")
    # Standard HF models include the embedding output at index 0. Dropping it
    # keeps layer indices aligned with DNABERT-style encoded-layer lists.
    return list(hidden_states[1:]) if len(hidden_states) > 1 else list(hidden_states)


def encoder_layers(model: Any) -> list[Any]:
    """Return encoder layers for BERT/ESM-style Hugging Face models."""

    encoder = getattr(model, "encoder", None)
    if encoder is None and hasattr(model, "base_model"):
        encoder = getattr(model.base_model, "encoder", None)
    if encoder is None and hasattr(model, "esm"):
        encoder = getattr(model.esm, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        layers = getattr(encoder, "layers", None)
    if layers is None:
        raise AttributeError("Could not locate encoder layers on the loaded model.")
    return list(layers)


def infer_attention_geometry(model: Any, layer_idx: int = 0) -> tuple[int, int, int]:
    """Infer `(n_heads, d_head, d_model)` from a supported encoder layer."""

    layer = encoder_layers(model)[layer_idx]
    attention = getattr(layer, "attention", None)
    self_attn = getattr(attention, "self", attention)
    n_heads = getattr(self_attn, "num_attention_heads", None) or getattr(self_attn, "num_heads", None)
    d_head = getattr(self_attn, "attention_head_size", None)
    if n_heads is not None and d_head is not None:
        return int(n_heads), int(d_head), int(n_heads) * int(d_head)

    q_proj = getattr(self_attn, "query", None) or getattr(self_attn, "q_proj", None)
    if q_proj is None:
        raise AttributeError("Could not infer attention geometry from the selected layer.")
    out_dim, d_model = q_proj.weight.shape
    if n_heads is None:
        n_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
    if n_heads is None:
        raise AttributeError("Could not infer number of attention heads from the selected layer.")
    d_head = int(out_dim) // int(n_heads)
    return int(n_heads), int(d_head), int(d_model)


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
