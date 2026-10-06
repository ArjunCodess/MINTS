"""Validated DNABERT native attention and nucleotide-level localization."""
from __future__ import annotations

import numpy as np
import torch

from .assay_alignment import exact_offsets, base_attention_density
from .modeling import encoder_layers, encode_sequences


def capture_localization(bundle, sequence, span, motif_scores=None):
    offsets = exact_offsets(bundle.tokenizer, sequence)
    real = [i for i, (a, b) in enumerate(offsets) if b > a]
    start, end = span
    target = [i for i in real if offsets[i][0] < end and offsets[i][1] > start]
    if not target:
        raise ValueError("Native localization span lacks token support")
    captured, handles = {}, []
    def make_hook(layer):
        def hook(module, inputs, output):
            hidden, cu, max_length, indices, mask, bias = inputs
            count = hidden.shape[0]
            if len(cu) != 2 or count != int(max_length) or count != len(offsets):
                raise ValueError("Native assay requires one unpadded sequence with exact offsets")
            qkv = module.Wqkv(hidden).reshape(count, 3, module.num_attention_heads, module.attention_head_size)
            q, k, v = qkv[:, 0].permute(1, 0, 2), qkv[:, 1].permute(1, 2, 0), qkv[:, 2].permute(1, 0, 2)
            content = q @ k / np.sqrt(module.attention_head_size)
            native = torch.softmax(content + bias[0, :, :count, :count], dim=-1)
            reconstructed = (native @ v).permute(1, 0, 2).reshape(count, -1)
            error = float((reconstructed - output).abs().max().item())
            if not torch.allclose(reconstructed, output, atol=2e-4, rtol=2e-4):
                raise ValueError(f"Native head-context reconstruction differs by {error}")
            a = native.detach().cpu().numpy()
            no_position = torch.softmax(content, dim=-1).detach().cpu().numpy()
            positional = torch.softmax(bias[0, :, :count, :count].expand_as(content), dim=-1).detach().cpu().numpy()
            local_span = (max(0, start - 30), min(len(sequence), end + 30))
            scores = dict(base_density=base_attention_density(a, offsets, span),
                          local_query_density=base_attention_density(a, offsets, span, local_span),
                          content_density=base_attention_density(no_position, offsets, span),
                          position_only_density=base_attention_density(positional, offsets, span),
                          token_density=a[:, real][:, :, target].mean(axis=(1, 2)) * len(real),
                          max_context_error=error)
            if motif_scores is not None:
                from tools.review_ctcf_controls import sequence_qk_correlation
                key_scores = content[:, real][:, :, real].mean(dim=1).detach().cpu().numpy()
                scores["qk_r"] = sequence_qk_correlation(key_scores, np.asarray(motif_scores)[real])
            captured[layer] = scores
        return hook
    try:
        for layer, module in enumerate(encoder_layers(bundle.hf_model)):
            handles.append(module.attention.self.register_forward_hook(make_hook(layer)))
        encoded = encode_sequences(bundle.tokenizer, sequence, bundle.device, max_length=None)
        if encoded["input_ids"].shape[-1] != len(offsets):
            raise ValueError("Model encoding truncated the native assay sequence")
        bundle.hf_model.eval()
        with torch.no_grad():
            bundle.hf_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                            output_all_encoded_layers=False)
    finally:
        for handle in handles:
            handle.remove()
    keys = [k for k in captured[0] if k != "max_context_error"]
    return {**{k: np.stack([captured[i][k] for i in range(len(captured))]) for k in keys},
            "max_context_error": max(c["max_context_error"] for c in captured.values()),
            "real_tokens": len(real), "target_tokens": len(target), "span": span}
