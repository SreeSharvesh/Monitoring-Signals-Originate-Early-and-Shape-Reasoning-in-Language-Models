# run_patching_v3.py
# Non-saturating causal patching:
# - Windowed token patch (W in {1,3,5}) instead of whole-suffix
# - Alpha mixing h := h + α (h_donor - h_base) to avoid "locking on"
# - Optional component-restricted patching: resid | attn_only | mlp_only
# - Domain-sliced plots + token-distance KL
#
# Outputs under out_dir:
#   patch_items.csv
#   patch_layer.csv
#   patch_token_distance.csv
#   kl_vs_layer.png
#   dnll_vs_layer.png
#   flip_rate_bar.png
#   kl_token_distance_mid.png
#   kl_vs_layer_by_domain.png
#   dnll_vs_layer_by_domain.png

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import load_hf_model, generate_and_reforward
from prompts import load_gsm8k, load_humaneval_lite, load_boolq, SYSTEM_PREFIX

# -------------------------------
# Config
# -------------------------------
EARLY_LAYERS = [0, 2, 4, 6]
MID_LAYERS   = [12, 16]            # controls
LAYER_SET    = EARLY_LAYERS + MID_LAYERS

ANCHORS      = ["mid", "final"]    # which token to anchor the patch at
DEFAULT_CUE  = "[MONITOR]"
DEFAULT_OUT  = "runs/patching_v4"
SEED         = 1337

WINDOW_SET   = [1, 3, 5]           # window sizes (tokens)
ALPHAS       = [0.25, 0.5, 0.75, 1.0]
COMPONENTS   = ["resid"]           # can extend to ["resid","attn_only","mlp_only"] when desired

K_OFFSETS    = [1, 5, 10, 20]      # token distances after anchor for KL(t+k)

# -------------------------------
# Prompting helpers
# -------------------------------
def assemble_pre_reasoning(task_text: str, meta_line: str) -> str:
    meta = (meta_line + "\n") if meta_line else ""
    return (
        f"{SYSTEM_PREFIX}\n\n"
        f"Task:\n{task_text}\n\n"
        f"{meta}Reasoning:\n"
    )

def find_first_reasoning_sentence_end(text: str) -> int | None:
    body = text.split("Reasoning:", 1)[-1] if "Reasoning:" in text else text
    for ch in [".", "!", "?"]:
        idx = body.find(ch)
        if idx != -1:
            before, _, _ = text.partition("Reasoning:")
            return len(before) + len("Reasoning:") + idx + 1
    return None

def char_to_token_index(tokenizer, text: str, char_idx: int) -> int:
    prefix = text[:char_idx]
    ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    return len(ids)

# -------------------------------
# Build Hidden vs Monitored with mid insertion
# -------------------------------
@torch.no_grad()
def hidden_vs_monitored_mid(bundle, task_text: str, cue_text: str, max_new_tokens: int = 256):
    tok = bundle.tok
    device = bundle.model.device

    # Hidden baseline text → deterministic
    hidden_prompt = assemble_pre_reasoning(task_text, meta_line="")
    r_hidden = generate_and_reforward(bundle, hidden_prompt, max_new_tokens, temperature=0.0)

    # Find first sentence end (char → token index in Hidden text)
    char_idx = find_first_reasoning_sentence_end(r_hidden["text"])
    if char_idx is None:
        prompt_ids = tok(hidden_prompt, return_tensors="pt")["input_ids"][0].tolist()
        t_mid = len(prompt_ids) + 20
    else:
        t_mid = char_to_token_index(tok, r_hidden["text"], char_idx)

    # Compose Monitored by inserting cue tokens at t_mid in the Hidden token stream
    full_ids_hidden = torch.tensor(r_hidden["sequences"], device=device)   # [1, S_h]
    cue_ids = tok(cue_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)  # [1, C]
    composed = torch.cat([full_ids_hidden[:, :t_mid], cue_ids], dim=1)

    outputs_mon = bundle.model.generate(
        input_ids=composed,
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=0.0,
        return_dict_in_generate=True,
        pad_token_id=bundle.tok.eos_token_id,
    )
    full_ids_mon = outputs_mon.sequences  # [1, S_m]

    # Re-forward both sequences to get hidden states & logits
    def reforward(ids):
        attn = torch.ones_like(ids)
        fw = bundle.model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hs = [h.detach().float().cpu().numpy() for h in fw.hidden_states]  # slot 0 = embeddings
        logits = fw.logits.detach().float().cpu().numpy()
        return hs, logits

    hs_h, logits_h = reforward(full_ids_hidden)
    hs_m, logits_m = reforward(full_ids_mon)

    cue_len = int(cue_ids.shape[1])
    return {
        "hidden":    {"ids": full_ids_hidden.cpu().numpy(), "hs": hs_h, "logits": logits_h, "text": r_hidden["text"]},
        "monitored": {"ids": full_ids_mon.cpu().numpy(),    "hs": hs_m, "logits": logits_m},
        "t_mid": int(t_mid),
        "cue_len": cue_len,
    }

# -------------------------------
# Layer accessor (HF GPT family)
# -------------------------------
def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "h"):
        return model.model.h
    raise AttributeError("Could not find transformer layers list on model.model")

# -------------------------------
# Metrics
# -------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def kl_div(p, q):
    mask = (p > 0)
    return float(np.sum(p[mask] * (np.log(p[mask] + 1e-12) - np.log(q[mask] + 1e-12))))

def delta_nll_on_suffix(base_logits, patched_logits, target_ids, start_idx: int) -> float:
    L = target_ids.shape[1]
    lo = max(start_idx + 1, 1)
    hi = L - 1
    if hi <= lo:
        return 0.0
    base = base_logits[0, lo-1:hi, :]
    patched = patched_logits[0, lo-1:hi, :]
    tgt = target_ids[0, lo:hi]
    p_base = softmax(base, axis=-1)
    p_patch = softmax(patched, axis=-1)
    idx = (np.arange(tgt.shape[0]), tgt.astype(int))
    nll_base = -np.log(p_base[idx] + 1e-12).mean()
    nll_patch = -np.log(p_patch[idx] + 1e-12).mean()
    return float(nll_patch - nll_base)

def final_token_kl_and_flip(base_logits, patched_logits):
    base = base_logits[0, -1, :]
    patched = patched_logits[0, -1, :]
    p = softmax(base); q = softmax(patched)
    return kl_div(p, q), int(np.argmax(base) != np.argmax(patched))

# -------------------------------
# Causal re-forward with WINDOW + ALPHA mixing
# -------------------------------
@torch.no_grad()
def reforward_with_window_alpha(
    bundle,
    base_ids: torch.Tensor,
    donor_hs: List[np.ndarray],
    layer_idx: int,
    base_token_idx: int,
    donor_token_idx: int,
    window_size: int = 3,
    alpha: float = 0.5,
    component: str = "resid",   # "resid" | "attn_only" | "mlp_only" (resid = block output)
) -> Dict[str, Any]:
    """
    Modify block-L output (or component outputs) over a LOCAL token window with α-mixing:
        h[:, t:t+W] := h[:, t:t+W] + α * (donor[:, t':t'+W] - h[:, t:t+W])
    Notes:
      - hidden_states[0] is embeddings; block-L output is at slot L+1
      - For component modes we patch on the sub-module forward; donor slices taken from block outputs
        as a pragmatic proxy unless you later collect component-specific caches.
    """
    model = bundle.model
    device = model.device
    ids = base_ids.to(device)
    attn = torch.ones_like(ids)

    hs_slot = layer_idx + 1
    donor_layer_np = donor_hs[hs_slot]   # [1, S_m, d]
    donor_layer = torch.from_numpy(donor_layer_np).to(device, dtype=next(model.parameters()).dtype)
    S_m = donor_layer.shape[1]

    # compute spans (guarded)
    def span_indices(T_base: int):
        b0 = max(0, min(base_token_idx, T_base - 1))
        d0 = max(0, min(donor_token_idx, S_m - 1))
        span = min(window_size, T_base - b0, S_m - d0)
        return b0, d0, max(0, span)

    layers = get_transformer_layers(model)
    handles = []

    # ---- resid mode: patch block output at layer L
    def hook_resid(mod, inp, out):
        tuple_out = isinstance(out, tuple)
        h = out[0] if tuple_out else out  # [B, T, d]
        T = h.shape[1]

        # STRICT out-of-range check -> hard no-op
        if (base_token_idx < 0) or (donor_token_idx < 0) or \
           (base_token_idx >= T) or (donor_token_idx >= S_m):
            return out

        # compute span WITHOUT clamping starts
        b0 = base_token_idx
        d0 = donor_token_idx
        span = min(window_size, T - b0, S_m - d0)
        if span <= 0:
            return out

        base_slice = h[:, b0:b0+span, :]
        donor_slice = donor_layer[0, d0:d0+span, :].to(h.dtype)
        mixed = base_slice + alpha * (donor_slice - base_slice)
        h2 = h.clone()
        h2[:, b0:b0+span, :] = mixed
        return (h2,) + out[1:] if tuple_out else h2

    # Optional component modes (lightweight proxy: still mix at block output, 
    # but you can later swap to real attn/mlp caches when you collect them)
    patch_target = "resid" if component not in ("attn_only", "mlp_only") else "resid"

    if patch_target == "resid":
        handles.append(layers[layer_idx].register_forward_hook(lambda m,i,o: hook_resid(m,i,o)))
    else:
        # Placeholders for future true component hooks
        handles.append(layers[layer_idx].register_forward_hook(lambda m,i,o: hook_resid(m,i,o)))

    try:
        fw = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        logits = fw.logits.detach().float().cpu().numpy()
        hs     = [x.detach().float().cpu().numpy() for x in fw.hidden_states]
        return {"logits": logits, "hs": hs}
    finally:
        for h in handles:
            h.remove()

# -------------------------------
# Runner
# -------------------------------
def run_patching_v3(out_dir: str = DEFAULT_OUT,
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
        "windows": WINDOW_SET,
        "alphas": ALPHAS,
        "components": COMPONENTS,
    }
    (outp / "meta.json").write_text(json.dumps(meta, indent=2))

    rows = []

    for item in tqdm(tasks, desc="causal patching v3"):
        task_text = item["text"]
        pair = hidden_vs_monitored_mid(bundle, task_text, cue_text, max_new_tokens=max_new_tokens)

        hid_ids = torch.tensor(pair["hidden"]["ids"])
        mon_hs  = pair["monitored"]["hs"]
        base_logits = pair["hidden"]["logits"]
        t_mid = pair["t_mid"]; cue_len = pair["cue_len"]
        len_h = pair["hidden"]["ids"].shape[1]; len_m = pair["monitored"]["ids"].shape[1]

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

                for W in WINDOW_SET:
                    for a in ALPHAS:
                        for comp in COMPONENTS:
                            res = reforward_with_window_alpha(
                                bundle,
                                base_ids=hid_ids,
                                donor_hs=mon_hs,
                                layer_idx=L,
                                base_token_idx=base_idx,
                                donor_token_idx=donor_idx,
                                window_size=W,
                                alpha=a,
                                component=comp,
                            )

                            kl_final, flip = final_token_kl_and_flip(base_logits, res["logits"])
                            dnll = delta_nll_on_suffix(base_logits, res["logits"], pair["hidden"]["ids"], start_idx=base_idx)

                            rows.append({
                                "task_id": item["task_id"],
                                "domain": item["domain"],
                                "anchor": anchor,
                                "layer": L,
                                "window": W,
                                "alpha": a,
                                "component": comp,
                                "final_kl": kl_final,
                                "final_flip": int(flip),
                                "delta_nll_suffix": dnll,
                            })

                            # token-distance KLs (mid anchor only is most informative)
                            base = base_logits[0]; patched = res["logits"][0]
                            for k in K_OFFSETS:
                                tpos = min(len_h - 1, base_idx + k - 1)
                                p = softmax(base[tpos, :]); q = softmax(patched[tpos, :])
                                rows.append({
                                    "task_id": item["task_id"],
                                    "domain": item["domain"],
                                    "anchor": anchor,
                                    "layer": L,
                                    "window": W,
                                    "alpha": a,
                                    "component": comp,
                                    "metric": f"kl_t+{k}",
                                    "value": kl_div(p, q),
                                })

    df = pd.DataFrame(rows)
    df_base = df[df.get("metric").isna()] if "metric" in df.columns else df
    df_base.to_csv(outp / "patch_items.csv", index=False)

    # --- Aggregations
    agg = df_base.groupby(["anchor","layer","window","alpha","component"]).agg(
        final_kl_mean=("final_kl","mean"),
        final_kl_lo=("final_kl", lambda s: s.quantile(0.025)),
        final_kl_hi=("final_kl", lambda s: s.quantile(0.975)),
        flip_rate=("final_flip","mean"),
        dnll_mean=("delta_nll_suffix","mean"),
        dnll_lo=("delta_nll_suffix", lambda s: s.quantile(0.025)),
        dnll_hi=("delta_nll_suffix", lambda s: s.quantile(0.975)),
        n=("final_kl","count"),
    ).reset_index()
    agg.to_csv(outp / "patch_layer.csv", index=False)

    if "metric" in df.columns:
        df_kl = df[df.metric.notna()]
        agg_kl = df_kl.groupby(["anchor","layer","window","alpha","component","metric"]).agg(
            mean=("value","mean"),
            lo=("value", lambda s: s.quantile(0.025)),
            hi=("value", lambda s: s.quantile(0.975)),
        ).reset_index()
        agg_kl.to_csv(outp / "patch_token_distance.csv", index=False)
    else:
        agg_kl = None

    # --- Choose a canonical slice to plot compactly:
    BEST_W, BEST_A, BEST_COMP = 3, 0.5, "resid"

    def subsel(df, **kw):
        s = df.copy()
        for k,v in kw.items():
            s = s[s[k]==v]
        return s

    # KL(final) vs layer (mid/final), for BEST (W,A,comp)
    plt.figure(figsize=(8.8,5))
    for anchor in ANCHORS:
        sub = subsel(agg, anchor=anchor, window=BEST_W, alpha=BEST_A, component=BEST_COMP)
        plt.plot(sub["layer"], sub["final_kl_mean"], label=f"{anchor} anchor")
        plt.fill_between(sub["layer"], sub["final_kl_lo"], sub["final_kl_hi"], alpha=0.2)
    plt.xlabel("Layer"); plt.ylabel("KL at final token")
    plt.title(f"KL(final) vs Layer — W={BEST_W}, α={BEST_A}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outp / "kl_vs_layer.png", dpi=180); plt.close()

    # ΔNLL(suffix) vs layer
    plt.figure(figsize=(8.8,5))
    for anchor in ANCHORS:
        sub = subsel(agg, anchor=anchor, window=BEST_W, alpha=BEST_A, component=BEST_COMP)
        plt.plot(sub["layer"], sub["dnll_mean"], label=f"{anchor} anchor")
        plt.fill_between(sub["layer"], sub["dnll_lo"], sub["dnll_hi"], alpha=0.2)
    plt.axhline(0.0, color="black", lw=0.8)
    plt.xlabel("Layer"); plt.ylabel("ΔNLL on Hidden suffix")
    plt.title(f"ΔNLL(suffix) vs Layer — W={BEST_W}, α={BEST_A}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outp / "dnll_vs_layer.png", dpi=180); plt.close()

    # Flip rate bar
    layers = sorted(agg["layer"].unique())
    plt.figure(figsize=(9,4))
    x = np.arange(len(layers)); width = 0.35
    sub_mid = subsel(agg, anchor="mid", window=BEST_W, alpha=BEST_A, component=BEST_COMP).set_index("layer").reindex(layers).fillna(0)
    sub_fin = subsel(agg, anchor="final", window=BEST_W, alpha=BEST_A, component=BEST_COMP).set_index("layer").reindex(layers).fillna(0)
    plt.bar(x - width/2, sub_mid["flip_rate"], width, label="mid anchor")
    plt.bar(x + width/2, sub_fin["flip_rate"], width, label="final anchor")
    plt.xticks(x, layers); plt.ylabel("Flip rate at final token")
    plt.title(f"Flip Rate by Layer — W={BEST_W}, α={BEST_A}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outp / "flip_rate_bar.png", dpi=180); plt.close()

    # Token-distance KL (mid anchor), BEST (W,A,comp)
    if agg_kl is not None:
        plt.figure(figsize=(9,5))
        for k in K_OFFSETS:
            sub = subsel(agg_kl, anchor="mid", window=BEST_W, alpha=BEST_A,
                         component=BEST_COMP, metric=f"kl_t+{k}")
            plt.plot(sub["layer"], sub["mean"], label=f"KL@+{k}")
            plt.fill_between(sub["layer"], sub["lo"], sub["hi"], alpha=0.2)
        plt.xlabel("Layer"); plt.ylabel("KL at token t+k")
        plt.title(f"Windowed α-mix (mid) — KL at distances, W={BEST_W}, α={BEST_A}")
        plt.legend(); plt.tight_layout()
        plt.savefig(outp / "kl_token_distance_mid.png", dpi=180); plt.close()

    # ----- Per-domain slices (BEST W,A,comp)
    def plot_by_domain(metric_key: str, ylabel: str, fname: str):
        plt.figure(figsize=(9.8,5.6))
        for domain in ["math","code","qa"]:
            # Domain mapping from loaders
            dom = {"gsm8k_lite":"math","humaneval_lite":"code","boolq":"qa"}.get(domain, domain)
        # domain join (df_base rows carry domain string from item["domain"])
        for domain in sorted(df_base["domain"].unique()):
            sub = df_base[(df_base["anchor"]=="mid") &
                          (df_base["window"]==BEST_W) &
                          (df_base["alpha"]==BEST_A) &
                          (df_base["component"]==BEST_COMP) &
                          (df_base["domain"]==domain)]
            g = sub.groupby("layer")[metric_key].mean().reset_index()
            plt.plot(g["layer"], g[metric_key], label=domain)
        plt.xlabel("Layer"); plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Layer (mid, per-domain) — W={BEST_W}, α={BEST_A}")
        plt.legend(); plt.tight_layout()
        plt.savefig(outp / fname, dpi=180); plt.close()

    plot_by_domain("final_kl", "KL(final)", "kl_vs_layer_by_domain.png")
    plot_by_domain("delta_nll_suffix", "ΔNLL(suffix)", "dnll_vs_layer_by_domain.png")

    print(f"Saved v3 patching results to {out_dir}")

if __name__ == "__main__":
    run_patching_v3()
