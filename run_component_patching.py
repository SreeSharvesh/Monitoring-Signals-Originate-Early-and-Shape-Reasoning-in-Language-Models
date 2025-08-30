# run_component_patching.py
# Component-level causal patching (windowed α-mix, layer-aligned caches)

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from prompts import load_gsm8k, load_humaneval_lite, load_boolq, SYSTEM_PREFIX
from models import load_hf_model
from run_patching_v3 import (
    hidden_vs_monitored_mid,
    get_transformer_layers,
    reforward_with_window_alpha,  # validated residual (block-output) patcher
)

plt.rcParams.update({"figure.dpi": 180})
torch.set_grad_enabled(False)

# -------------------------------
# Config
# -------------------------------
EARLY_LAYERS = [0, 2, 4, 6]
MID_LAYERS   = [12, 16]  # controls
LAYER_SET    = EARLY_LAYERS + MID_LAYERS

ANCHORS      = ["mid", "final"]
DEFAULT_CUE  = "[MONITOR]"
DEFAULT_OUT  = "runs/component_v3"
SEED         = 1337

BEST_W       = 5
BEST_A       = 0.75
K_OFFSETS    = [1, 5, 10]

# Toggle a tiny amount of debug logging (first N items)
DEBUG_N = 0  # set to >0 to capture a few debug rows
DEBUG_ROWS: List[dict] = []

# -------------------------------
# Metrics
# -------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def kl_div(p, q):
    m = (p > 0)
    return float(np.sum(p[m] * (np.log(p[m] + 1e-12) - np.log(q[m] + 1e-12))))

def final_token_kl_and_flip(base_logits, patched_logits):
    base = base_logits[0, -1, :]
    patched = patched_logits[0, -1, :]
    p = softmax(base); q = softmax(patched)
    return kl_div(p, q), int(np.argmax(base) != np.argmax(patched))

def delta_nll_on_suffix(base_logits, patched_logits, target_ids, start_idx: int) -> float:
    L = target_ids.shape[1]
    lo = max(start_idx + 1, 1); hi = L - 1
    if hi <= lo:
        return 0.0
    base = base_logits[0, lo-1:hi, :]
    patc = patched_logits[0, lo-1:hi, :]
    tgt  = target_ids[0, lo:hi].astype(int)
    p_base = softmax(base, axis=-1)
    p_patc = softmax(patc, axis=-1)
    idx = (np.arange(tgt.shape[0]), tgt)
    return float((-np.log(p_patc[idx] + 1e-12)).mean() - (-np.log(p_base[idx] + 1e-12)).mean())

# -------------------------------
# Component cache collection (Monitored donor)
# -------------------------------
@torch.no_grad()
def collect_component_caches(bundle, ids: torch.Tensor) -> Dict[str, List[np.ndarray | None]]:
    """
    Run the model with forward hooks on each layer's components and
    collect *layer-aligned* outputs:
      - self_attn forward output (after o_proj etc.)
      - mlp forward output
    Returns lists of length n_layers; entries may be None if a submodule is absent.
    """
    model = bundle.model
    dev = model.device
    ids = ids.to(dev)
    attn_mask = torch.ones_like(ids)

    layers = get_transformer_layers(model)
    L = len(layers)
    store_attn: List[np.ndarray | None] = [None] * L
    store_mlp:  List[np.ndarray | None] = [None] * L
    handles = []

    def make_store(slot_list: List, idx: int):
        def _hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            slot_list[idx] = h.detach().float().cpu().numpy()
            return None  # do not modify the forward
        return _hook

    for i, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            handles.append(layer.self_attn.register_forward_hook(make_store(store_attn, i)))
        if hasattr(layer, "mlp"):
            handles.append(layer.mlp.register_forward_hook(make_store(store_mlp, i)))

    try:
        _ = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=False, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {"attn": store_attn, "mlp": store_mlp}

# -------------------------------
# Component-only patching (true submodule hooks)
# -------------------------------
@torch.no_grad()
def reforward_with_component_alpha_window(
    bundle,
    base_ids: torch.Tensor,
    donor_comp: List[np.ndarray | None],  # layer-aligned; item may be None
    layer_idx: int,
    base_token_idx: int,
    donor_token_idx: int,
    window_size: int = BEST_W,
    alpha: float = BEST_A,
    component: str = "attn",  # "attn" | "mlp"
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Mix ONLY the chosen component's output at the given layer over a local window:
        h[:, t:t+W] ← h[:, t:t+W] + α ( donor[:, t':t'+W] - h[:, t:t+W] )
    If donor_comp[layer_idx] is None, performs a clean no-op forward.
    """
    assert component in ("attn", "mlp")
    model = bundle.model
    dev = model.device
    ids = base_ids.to(dev)
    attn_mask = torch.ones_like(ids)

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    if layer_idx >= n_layers:
        fw = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=False, use_cache=False)
        return {"logits": fw.logits.detach().float().cpu().numpy()}

    donor_np = donor_comp[layer_idx]
    if donor_np is None:
        fw = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=False, use_cache=False)
        return {"logits": fw.logits.detach().float().cpu().numpy()}

    donor_layer = torch.from_numpy(donor_np).to(dev, dtype=next(model.parameters()).dtype)
    S_m = donor_layer.shape[1]

    def patch_hook(mod, inp, out):
        tuple_out = isinstance(out, tuple)
        h = out[0] if tuple_out else out  # [B,T,d]
        T = h.shape[1]

        if (base_token_idx < 0) or (donor_token_idx < 0) or (base_token_idx >= T) or (donor_token_idx >= S_m):
            return None if tuple_out else out

        b0 = base_token_idx
        d0 = donor_token_idx
        span = min(window_size, T - b0, S_m - d0)
        if span <= 0:
            return None if tuple_out else out

        base_slice  = h[:, b0:b0+span, :]
        donor_slice = donor_layer[0, d0:d0+span, :].to(h.dtype)
        mixed = base_slice + float(alpha) * (donor_slice - base_slice)

        if debug:
            DEBUG_ROWS.append({
                "layer": layer_idx, "component": component,
                "base_idx": base_token_idx, "donor_idx": donor_token_idx,
                "W": span, "alpha": float(alpha),
                "base_slice_meanabs": float(base_slice.abs().mean()),
                "donor_slice_meanabs": float(donor_slice.abs().mean()),
                "delta_meanabs": float((donor_slice - base_slice).abs().mean()),
            })

        h2 = h.clone()
        h2[:, b0:b0+span, :] = mixed
        return (h2,) + out[1:] if tuple_out else h2

    target = layers[layer_idx].self_attn if component == "attn" else layers[layer_idx].mlp
    handle = target.register_forward_hook(patch_hook)
    try:
        fw = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=False, use_cache=False)
        return {"logits": fw.logits.detach().float().cpu().numpy()}
    finally:
        handle.remove()

# -------------------------------
# Runner
# -------------------------------
def run_component_patching(out_dir: str = DEFAULT_OUT,
                           model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                           n_math: int = 20, n_code: int = 20, n_qa: int = 20,
                           max_new_tokens: int = 256, seed: int = SEED,
                           cue_text: str = DEFAULT_CUE):
    rng = np.random.RandomState(seed)
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    bundle = load_hf_model(model_name)
    tasks = load_gsm8k(n_math) + load_humaneval_lite(n_code) + load_boolq(n_qa)
    rng.shuffle(tasks)

    meta = {
        "model": model_name,
        "n_items": len(tasks),
        "layers": LAYER_SET,
        "anchors": ANCHORS,
        "cue_text": cue_text,
        "seed": seed,
        "window": BEST_W,
        "alpha": BEST_A,
    }
    (outp / "meta.json").write_text(json.dumps(meta, indent=2))

    rows: List[dict] = []
    dbg_budget = DEBUG_N

    for item in tqdm(tasks, desc="component-level patching"):
        pair = hidden_vs_monitored_mid(bundle, item["text"], cue_text, max_new_tokens=max_new_tokens)

        hid_ids     = torch.tensor(pair["hidden"]["ids"])
        base_logits = pair["hidden"]["logits"]
        t_mid, cue_len = int(pair["t_mid"]), int(pair["cue_len"])
        len_h = pair["hidden"]["ids"].shape[1]
        len_m = pair["monitored"]["ids"].shape[1]

        # Donor component caches (Monitored), layer-aligned lists (may contain None)
        mon_ids = torch.tensor(pair["monitored"]["ids"])
        donor_comp = collect_component_caches(bundle, mon_ids)

        for anchor in ANCHORS:
            if anchor == "mid":
                base_idx  = t_mid
                donor_idx = t_mid + cue_len
            else:
                base_idx  = len_h - 1
                donor_idx = min(len_h, len_m) - 1

            if not (0 <= base_idx < len_h and 0 <= donor_idx < len_m):
                continue

            for L in LAYER_SET:
                n_layers = len(get_transformer_layers(bundle.model))
                if L >= n_layers:
                    continue

                # Residual (full block output) using validated patcher
                res_resid = reforward_with_window_alpha(
                    bundle,
                    base_ids=hid_ids,
                    donor_hs=pair["monitored"]["hs"],
                    layer_idx=L,
                    base_token_idx=base_idx,
                    donor_token_idx=donor_idx,
                    window_size=BEST_W,
                    alpha=BEST_A,
                    component="resid",
                )
                klR, flipR = final_token_kl_and_flip(base_logits, res_resid["logits"])
                dnllR = delta_nll_on_suffix(base_logits, res_resid["logits"], pair["hidden"]["ids"], base_idx)
                rows.append({
                    "task_id": item["task_id"], "domain": item["domain"],
                    "anchor": anchor, "layer": L, "component": "resid",
                    "final_kl": klR, "final_flip": int(flipR), "delta_nll_suffix": dnllR,
                })

                # Attention-only (true component)
                res_attn = reforward_with_component_alpha_window(
                    bundle, hid_ids, donor_comp["attn"],
                    layer_idx=L, base_token_idx=base_idx, donor_token_idx=donor_idx,
                    window_size=BEST_W, alpha=BEST_A,
                    component="attn", debug=(dbg_budget > 0),
                )
                klA, flipA = final_token_kl_and_flip(base_logits, res_attn["logits"])
                dnllA = delta_nll_on_suffix(base_logits, res_attn["logits"], pair["hidden"]["ids"], base_idx)
                rows.append({
                    "task_id": item["task_id"], "domain": item["domain"],
                    "anchor": anchor, "layer": L, "component": "attn_only",
                    "final_kl": klA, "final_flip": int(flipA), "delta_nll_suffix": dnllA,
                })

                # MLP-only (true component)
                res_mlp = reforward_with_component_alpha_window(
                    bundle, hid_ids, donor_comp["mlp"],
                    layer_idx=L, base_token_idx=base_idx, donor_token_idx=donor_idx,
                    window_size=BEST_W, alpha=BEST_A,
                    component="mlp", debug=(dbg_budget > 0),
                )
                klM, flipM = final_token_kl_and_flip(base_logits, res_mlp["logits"])
                dnllM = delta_nll_on_suffix(base_logits, res_mlp["logits"], pair["hidden"]["ids"], base_idx)
                rows.append({
                    "task_id": item["task_id"], "domain": item["domain"],
                    "anchor": anchor, "layer": L, "component": "mlp_only",
                    "final_kl": klM, "final_flip": int(flipM), "delta_nll_suffix": dnllM,
                })

                # Token-distance KL (mid anchor only)
                if anchor == "mid":
                    baseL = base_logits[0]
                    for comp_name, logits in [("resid", res_resid["logits"]),
                                              ("attn_only", res_attn["logits"]),
                                              ("mlp_only", res_mlp["logits"])]:
                        patcL = logits[0]
                        for k in K_OFFSETS:
                            # logits at t-1 predict token t
                            tpos = min(len_h - 1, base_idx + k - 1)
                            p = softmax(baseL[tpos, :]); q = softmax(patcL[tpos, :])
                            rows.append({
                                "task_id": item["task_id"], "domain": item["domain"],
                                "anchor": anchor, "layer": L, "component": comp_name,
                                "metric": f"kl_t+{k}", "value": kl_div(p, q),
                            })

            if dbg_budget > 0:
                dbg_budget -= 1  # only record a few items

    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    # Debug CSV (optional)
    if DEBUG_ROWS:
        pd.DataFrame(DEBUG_ROWS).to_csv(outp / "component_debug.csv", index=False)

    # Main outputs
    df = pd.DataFrame(rows)
    df_items = df[df.get("metric").isna()] if "metric" in df.columns else df
    df_items.to_csv(outp / "component_patch_items.csv", index=False)

    # Aggregates
    agg = df_items.groupby(["anchor","layer","component"]).agg(
        final_kl_mean=("final_kl","mean"),
        final_kl_lo=("final_kl", lambda s: s.quantile(0.025)),
        final_kl_hi=("final_kl", lambda s: s.quantile(0.975)),
        flip_rate=("final_flip","mean"),
        dnll_mean=("delta_nll_suffix","mean"),
        dnll_lo=("delta_nll_suffix", lambda s: s.quantile(0.025)),
        dnll_hi=("delta_nll_suffix", lambda s: s.quantile(0.975)),
        n=("final_kl","count"),
    ).reset_index()
    agg.to_csv(outp / "component_patch_layer.csv", index=False)

    # Effect shares vs residual
    shares = []
    for (anchor, layer), sub in agg.groupby(["anchor","layer"]):
        rowR = sub[sub.component == "resid"]
        if len(rowR) == 0:
            continue
        base_dnll = float(rowR["dnll_mean"].iloc[0])
        for comp in ["attn_only", "mlp_only"]:
            rowC = sub[sub.component == comp]
            if len(rowC) == 0:
                continue
            share = float(rowC["dnll_mean"].iloc[0]) / base_dnll if abs(base_dnll) > 1e-12 else np.nan
            shares.append({"anchor": anchor, "layer": layer, "component": comp, "dnll_share_of_resid": share})
    df_share = pd.DataFrame(shares)
    df_share.to_csv(outp / "component_share.csv", index=False)

    # Token-distance aggregates
    if "metric" in df.columns:
        df_kl = df[df.metric.notna()]
        agg_kl = df_kl.groupby(["anchor","layer","component","metric"]).agg(
            mean=("value","mean"),
            lo=("value", lambda s: s.quantile(0.025)),
            hi=("value", lambda s: s.quantile(0.975)),
        ).reset_index()
        agg_kl.to_csv(outp / "component_token_distance.csv", index=False)
    else:
        agg_kl = None

    # ---------------- Plots ----------------
    def plot_line_ci(ax, sub, x="layer", y="dnll_mean", lo="dnll_lo", hi="dnll_hi", label=None):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], label=label)
        ax.fill_between(sub[x], sub[lo], sub[hi], alpha=0.2)

    comps_order = ["resid", "attn_only", "mlp_only"]
    comp_names  = {"resid":"Residual (full)", "attn_only":"Attention-only", "mlp_only":"MLP-only"}

    # ΔNLL vs layer
    fig, ax = plt.subplots(figsize=(9,5))
    for comp in comps_order:
        sub = agg[(agg.anchor=="mid") & (agg.component==comp)]
        if len(sub): plot_line_ci(ax, sub, y="dnll_mean", lo="dnll_lo", hi="dnll_hi", label=comp_names[comp])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("ΔNLL on Hidden suffix")
    ax.set_title(f"Component-level ΔNLL vs Layer — W={BEST_W}, α={BEST_A}, anchor=mid")
    ax.legend(); fig.tight_layout(); fig.savefig(outp/"component_dnll_vs_layer.png"); plt.close(fig)

    # KL(final) vs layer
    fig, ax = plt.subplots(figsize=(9,5))
    for comp in comps_order:
        sub = agg[(agg.anchor=="mid") & (agg.component==comp)]
        if len(sub): plot_line_ci(ax, sub, y="final_kl_mean", lo="final_kl_lo", hi="final_kl_hi", label=comp_names[comp])
    ax.set_xlabel("Layer"); ax.set_ylabel("KL at final token")
    ax.set_title(f"Component-level KL(final) vs Layer — W={BEST_W}, α={BEST_A}, anchor=mid")
    ax.legend(); fig.tight_layout(); fig.savefig(outp/"component_kl_vs_layer.png"); plt.close(fig)

    # Flip-rate by layer
    fig, ax = plt.subplots(figsize=(10,4.8))
    layers = sorted(agg.layer.unique())
    width = 0.25
    xs = np.arange(len(layers))
    for j, comp in enumerate(comps_order):
        sub = agg[(agg.anchor=="mid") & (agg.component==comp)].set_index("layer").reindex(layers).fillna(0)
        ax.bar(xs + (j-1)*width, sub["flip_rate"].values, width, label=comp_names[comp])
    ax.set_xticks(xs); ax.set_xticklabels(layers)
    ax.set_ylabel("Flip rate at final token"); ax.set_xlabel("Layer")
    ax.set_title(f"Component-level Final-token Flip Rate — W={BEST_W}, α={BEST_A}, anchor=mid")
    ax.legend(); fig.tight_layout(); fig.savefig(outp/"component_flip_rate_by_layer.png"); plt.close(fig)

    # Share of residual effect (ΔNLL)
    fig, ax = plt.subplots(figsize=(9,5))
    for comp in ["attn_only","mlp_only"]:
        sub = df_share[(df_share.anchor=="mid") & (df_share.component==comp)].sort_values("layer")
        if len(sub): ax.plot(sub["layer"], sub["dnll_share_of_resid"], marker="o", label=comp_names[comp])
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Layer"); ax.set_ylabel("ΔNLL fraction of Residual")
    ax.set_title(f"Component share of full effect (ΔNLL) — W={BEST_W}, α={BEST_A}, anchor=mid")
    ax.legend(); fig.tight_layout(); fig.savefig(outp/"component_share_vs_layer.png"); plt.close(fig)

    # Token-distance KL (mid)
    if agg_kl is not None:
        for comp in ["attn_only","mlp_only"]:
            fig, ax = plt.subplots(figsize=(9,5))
            for k in K_OFFSETS:
                sub = agg_kl[(agg_kl.anchor=="mid") & (agg_kl.component==comp) & (agg_kl.metric==f"kl_t+{k}")]
                if len(sub): plot_line_ci(ax, sub, y="mean", lo="lo", hi="hi", label=f"KL@+{k}")
            ax.set_xlabel("Layer"); ax.set_ylabel("KL at token t+k")
            ax.set_title(f"{comp_names[comp]}: KL by token distance — W={BEST_W}, α={BEST_A}, anchor=mid")
            ax.legend(); fig.tight_layout(); fig.savefig(outp/f"component_token_distance_mid_{comp}.png"); plt.close(fig)

    print(f"Saved component-level results to {out_dir}")

if __name__ == "__main__":
    run_component_patching()
