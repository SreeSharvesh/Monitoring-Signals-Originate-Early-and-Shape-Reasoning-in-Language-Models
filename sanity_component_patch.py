# src/debug_component_patching.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from models import load_hf_model, generate_and_reforward
from prompts import SYSTEM_PREFIX, load_gsm8k

torch.set_grad_enabled(False)

# ---------- small utils ----------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def kl_div(p, q):
    m = (p > 0)
    return float(np.sum(p[m] * (np.log(p[m] + 1e-12) - np.log(q[m] + 1e-12))))

def final_token_kl_and_flip(base_logits, patched_logits):
    p = softmax(base_logits[0, -1, :]); q = softmax(patched_logits[0, -1, :])
    return kl_div(p, q), int(np.argmax(p) != np.argmax(q))

def get_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "h"):
        return model.model.h
    raise AttributeError("Cannot find transformer layers list on model.model")

def assemble_pre_reasoning(task_text: str, meta_line: str="") -> str:
    meta = (meta_line + "\n") if meta_line else ""
    return f"{SYSTEM_PREFIX}\n\nTask:\n{task_text}\n\n{meta}Reasoning:\n"

def find_first_reasoning_sentence_end(text: str) -> int | None:
    body = text.split("Reasoning:", 1)[-1] if "Reasoning:" in text else text
    for ch in [".", "!", "?"]:
        idx = body.find(ch)
        if idx != -1:
            before, _, _ = text.partition("Reasoning:")
            return len(before) + len("Reasoning:") + idx + 1
    return None

def char_to_token_index(tok, text: str, char_idx: int) -> int:
    prefix = text[:char_idx]
    ids = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    return len(ids)

# ---------- donor builder ----------
@torch.no_grad()
def build_hidden_and_monitored(bundle, task_text: str, cue_text="[MONITOR]", max_new_tokens=128):
    tok = bundle.tok; model = bundle.model; device = model.device

    prompt_hidden = assemble_pre_reasoning(task_text, "")
    r_hidden = generate_and_reforward(bundle, prompt_hidden, max_new_tokens, temperature=0.0)

    char_idx = find_first_reasoning_sentence_end(r_hidden["text"])
    if char_idx is None:
        t_mid = len(tok(prompt_hidden, return_tensors="pt")["input_ids"][0]) + 20
    else:
        t_mid = char_to_token_index(tok, r_hidden["text"], char_idx)

    ids_hidden = torch.tensor(r_hidden["sequences"], device=device)         # [1, S_h]
    cue_ids = tok(cue_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)  # [1,C]
    ids_monitored_prefix = torch.cat([ids_hidden[:, :t_mid], cue_ids], dim=1)

    out_mon = model.generate(
        input_ids=ids_monitored_prefix,
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=0.0,
        return_dict_in_generate=True,
        pad_token_id=bundle.tok.eos_token_id,
    )
    ids_monitored = out_mon.sequences

    attn = torch.ones_like(ids_hidden)
    fw_h = model(input_ids=ids_hidden, attention_mask=attn, output_hidden_states=True, use_cache=False)
    fw_m = model(input_ids=ids_monitored, attention_mask=torch.ones_like(ids_monitored),
                 output_hidden_states=True, use_cache=False)

    return {
        "ids_hidden": ids_hidden,
        "ids_monitored": ids_monitored,
        "logits_hidden": fw_h.logits.detach().cpu().numpy(),
        "logits_monitored": fw_m.logits.detach().cpu().numpy(),
        "hs_hidden": [h.detach().cpu() for h in fw_h.hidden_states],        # slot 0 = embeddings
        "hs_monitored": [h.detach().cpu() for h in fw_m.hidden_states],
        "t_mid": int(t_mid),
        "cue_len": int(cue_ids.shape[1]),
        "text_hidden": r_hidden["text"],
    }

# ---------- component cache collection ----------
@torch.no_grad()
def collect_component_caches(model, ids: torch.Tensor, layer_idxs: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Returns per-layer dict with keys:
      - 'attn': output of self_attn BEFORE residual add  (shape [B,T,d])
      - 'mlp' : output of mlp BEFORE residual add       (shape [B,T,d])
      - 'resid': layer output hidden_states AFTER both adds (h_L)         (shape [B,T,d])
    """
    layers = get_layers(model)
    want = set(layer_idxs)
    caches: Dict[int, Dict[str, torch.Tensor]] = {L:{} for L in want}

    handles = []

    def mk_attn_hook(L):
        def hook(mod, inp, out):
            # 'out' should be [B,T,d] or a tuple whose first is [B,T,d]
            o = out[0] if isinstance(out, tuple) else out
            caches[L]["attn"] = o.detach().clone()
            return out
        return hook

    def mk_mlp_hook(L):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caches[L]["mlp"] = o.detach().clone()
            return out
        return hook

    def mk_layer_out_hook(L):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caches[L]["resid"] = o.detach().clone()
            return out
        return hook

    for L, block in enumerate(layers):
        if L in want:
            # Qwen2/LLaMA-style names: .self_attn or .attention, and .mlp or .feed_forward
            attn_mod = getattr(block, "self_attn", getattr(block, "attention", None))
            mlp_mod  = getattr(block, "mlp", getattr(block, "feed_forward", None))
            assert attn_mod is not None and mlp_mod is not None, \
                f"Could not find attention/mlp modules on layer {L}"

            handles.append(attn_mod.register_forward_hook(mk_attn_hook(L)))
            handles.append(mlp_mod.register_forward_hook(mk_mlp_hook(L)))
            handles.append(block.register_forward_hook(mk_layer_out_hook(L)))

    try:
        _ = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return caches

# ---------- patch runner with DEBUG ----------
@torch.no_grad()
def run_with_component_patch_DEBUG(bundle,
                                   base_ids: torch.Tensor,
                                   donor_caches: Dict[int, Dict[str, torch.Tensor]],
                                   layer_idx: int,
                                   base_token_idx: int,
                                   donor_token_idx: int,
                                   window_size: int,
                                   alpha: float,
                                   component: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    component in {'resid','attn','mlp'} — we modify the COMPONENT OUTPUT at layer L,
    over [base_token_idx : base_token_idx+W), α-mixing with donor slices aligned at donor_token_idx.
    """
    model = bundle.model
    device = model.device
    dtype = next(model.parameters()).dtype
    layers = get_layers(model)

    ids = base_ids.to(device)
    attn = torch.ones_like(ids)

    donor_map = donor_caches[layer_idx]
    assert component in donor_map, f"Donor cache missing '{component}' for layer {layer_idx}"
    donor_L: torch.Tensor = donor_map[component].to(device=device, dtype=dtype)   # [B,Tm,d]
    Sm = donor_L.shape[1]

    debug = {"layer": layer_idx, "component": component, "alpha": float(alpha),
             "base_idx": int(base_token_idx), "donor_idx": int(donor_token_idx),
             "window": int(window_size), "hook_fired": 0,
             "base_slice_norm": 0.0, "donor_slice_norm": 0.0, "delta_norm": 0.0}

    def span(Tb: int):
        b0 = max(0, min(base_token_idx, Tb - 1))
        d0 = max(0, min(donor_token_idx, Sm - 1))
        W = min(window_size, Tb - b0, Sm - d0)
        return b0, d0, max(0, W)

    handles = []

    # ---- hook creators
    def make_component_hook(mod_name: str):
        def hook(mod, inp, out):
            # We tap the SUBMODULE output (attn or mlp), or the whole layer (resid)
            tuple_out = isinstance(out, tuple)
            h = out[0] if tuple_out else out  # [B,T,d]
            B,T,D = h.shape
            b0, d0, W = span(T)
            if W <= 0 or alpha == 0.0:
                return out

            base_slice = h[:, b0:b0+W, :]
            donor_slice = donor_L[:, d0:d0+W, :].to(h.dtype)

            mixed = base_slice + alpha * (donor_slice - base_slice)
            h2 = h.clone()
            h2[:, b0:b0+W, :] = mixed

            # DEBUG stats
            debug["hook_fired"] += 1
            debug["base_slice_norm"] = float(base_slice.norm().detach().cpu())
            debug["donor_slice_norm"] = float(donor_slice.norm().detach().cpu())
            debug["delta_norm"] = float((mixed - base_slice).norm().detach().cpu())

            return (h2,) + out[1:] if tuple_out else h2
        return hook

    # pick module to hook
    block = layers[layer_idx]
    if component == "attn":
        attn_mod = getattr(block, "self_attn", getattr(block, "attention", None))
        assert attn_mod is not None, f"No attention module on layer {layer_idx}"
        handles.append(attn_mod.register_forward_hook(make_component_hook("attn")))
    elif component == "mlp":
        mlp_mod = getattr(block, "mlp", getattr(block, "feed_forward", None))
        assert mlp_mod is not None, f"No MLP module on layer {layer_idx}"
        handles.append(mlp_mod.register_forward_hook(make_component_hook("mlp")))
    else:  # resid
        handles.append(block.register_forward_hook(make_component_hook("resid")))

    try:
        fw = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        logits = fw.logits.detach().float().cpu().numpy()
        hs     = [h.detach().float().cpu().numpy() for h in fw.hidden_states]
        return {"logits": logits, "hs": hs}, debug
    finally:
        for h in handles:
            h.remove()

# ---------- one-shot debug runner ----------
def debug_one_item(layer_idx=0, component="attn", W=5, alpha=0.75, cue="[MONITOR]"):
    bundle = load_hf_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    item = load_gsm8k(1)[0]
    pair = build_hidden_and_monitored(bundle, item["text"], cue_text=cue, max_new_tokens=128)

    ids_h = pair["ids_hidden"]
    ids_m = pair["ids_monitored"]
    t_mid = pair["t_mid"]; cue_len = pair["cue_len"]
    base_idx = t_mid
    donor_idx = t_mid + cue_len

    # 1) collect donor component caches at the desired layer(s)
    caches = collect_component_caches(bundle.model, ids_m, [layer_idx])

    # 2) base logits (for comparison / KL)
    base_logits = pair["logits_hidden"]

    # 3) run patch with DEBUG
    patched, dbg = run_with_component_patch_DEBUG(
        bundle, ids_h, caches,
        layer_idx=layer_idx,
        base_token_idx=base_idx,
        donor_token_idx=donor_idx,
        window_size=W,
        alpha=alpha,
        component=component,
    )

    kl, flip = final_token_kl_and_flip(base_logits, patched["logits"])

    # 4) print debug summary
    print("\n=== DEBUG COMPONENT PATCH ===")
    print(f"layer={layer_idx}  component={component}  W={W}  α={alpha}")
    print(f"base_idx={base_idx} donor_idx={donor_idx}  hook_fired={dbg['hook_fired']}")
    print(f"base_slice_norm={dbg['base_slice_norm']:.4f}  donor_slice_norm={dbg['donor_slice_norm']:.4f}  Δnorm={dbg['delta_norm']:.4f}")
    print(f"KL(final)={kl:.6f}  flip={flip}")

    # 5) invariants (assertions you can comment in/out)
    if alpha == 0.0 or W <= 0:
        assert dbg["hook_fired"] in (0, 1), "Hook should not change state when α=0 or W<=0"
        assert kl < 1e-6, "No-op should leave logits almost unchanged"
    else:
        assert dbg["hook_fired"] == 1, "Your component hook did not fire exactly once"
        assert dbg["delta_norm"] > 0.0, "Slice did not change — hook likely not applied where intended"

if __name__ == "__main__":
    # Try a few variants to sanity-check the hooks:
    debug_one_item(layer_idx=0, component="attn", W=5, alpha=0.75)   # early attention
    debug_one_item(layer_idx=2, component="attn", W=5, alpha=0.75)
    debug_one_item(layer_idx=6, component="mlp",  W=5, alpha=0.75)   # mid MLP
    debug_one_item(layer_idx=0, component="resid",W=5, alpha=0.50)   # residual baseline
