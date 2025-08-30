
# patching.py
# Residual/hidden-state cross-patching for HF models at a chosen layer and token index.
# We intercept the output of a transformer block and replace a SINGLE TOKEN vector with a donor vector.

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM

def _get_layers_module(model: AutoModelForCausalLM):
    """
    Try common paths to the list of decoder layers.
    Qwen2/LLaMA families: model.model.layers
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not locate transformer layers list on this model.")

def cross_patch_single_token(model: AutoModelForCausalLM,
                             input_ids: torch.Tensor,
                             donor_hidden_states: List[torch.Tensor],
                             layer_index: int,
                             token_index: int) -> Dict[str, Any]:
    """
    Perform a single forward pass where we replace the output hidden state at (layer_index, token_index)
    with a donor vector from donor_hidden_states[layer_index][0, token_index, :].

    Returns the re-decoded text and logits (optional). Assumes batch size 1.
    """
    print("In cross patch single token")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    layers = _get_layers_module(model)
    assert 0 <= layer_index < len(layers), "layer index out of range"

    donor_vec = donor_hidden_states[layer_index][0, token_index, :]  # numpy or tensor
    if not torch.is_tensor(donor_vec):
        donor_vec = torch.tensor(donor_vec, dtype=next(model.parameters()).dtype, device=device)
    else:
        donor_vec = donor_vec.to(device)

    handle = None
    def hook_fn(module, inputs, output):
        # output is hidden_states tensor of shape [B, T, d]
        # We replace ONLY at token_index for B=1
        out = output.clone()
        out[:, token_index, :] = donor_vec
        return out

    handle = layers[layer_index].register_forward_hook(lambda m, inp, out: hook_fn(m, inp, out))
    try:
        fw = model(input_ids=input_ids, output_hidden_states=False, use_cache=False)
        return {"last_hidden_state": fw.last_hidden_state.detach().cpu()}
    finally:
        if handle is not None:
            handle.remove()
