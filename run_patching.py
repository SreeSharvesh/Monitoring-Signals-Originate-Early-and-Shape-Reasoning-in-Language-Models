# run_patching.py
# Step 3: Causal patching for Monitoring Awareness
#
# What it does
# ------------
# 1) Builds Hidden vs Monitored (compact cue "[MONITOR]") with MID-REASONING insertion.
# 2) Collects full-sequence hidden states for both runs.
# 3) Causal patching: re-forwards the Hidden sequence while overwriting the
#    output at specified LAYERS and TOKEN ANCHORS with donor (Monitored) vectors.
#    We then read out:
#       - ΔNLL on the suffix (t >= anchor) of the *observed Hidden tokens*
#       - KL divergence at the final token
#       - Argmax flip rate at the final token
# 4) Writes clean CSVs + plots.
#
# Outputs (out_dir):
#   patch_items.csv      # per-item x (layer, anchor) metrics
#   patch_layer.csv      # per-layer means (anchor split)
#   kl_vs_layer.png      # KL(final-token) vs layer (for both anchors)
#   dnll_vs_layer.png    # ΔNLL(suffix) vs layer (for both anchors)
#   flip_rate_bar.png    # argmax flip rate at final token vs layer (for both anchors)
#
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

from models import load_hf_model, generate_and_reforward
from prompts import load_gsm8k, load_humaneval_lite, load_boolq, SYSTEM_PREFIX
from divergence import layerwise_cosine_distance

# -------------------------------
# Config
# -------------------------------
EARLY_LAYERS = [0, 2, 4, 6]
MID_LAYERS   = [12, 16]      # controls
LAYER_SET    = EARLY_LAYERS + MID_LAYERS

ANCHORS = ["mid", "final"]   # token index to patch at
DEFAULT_CUE = "[MONITOR]"
DEFAULT_OUT = "runs/patching_v2"
SEED = 1337

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

    # Hidden baseline text → generate deterministic
    hidden_prompt = assemble_pre_reasoning(task_text, meta_line="")
    r_hidden = generate_and_reforward(bundle, hidden_prompt, max_new_tokens, temperature=0.0)

    # Find first sentence end (char → token index in *Hidden* text)
    char_idx = find_first_reasoning_sentence_end(r_hidden["text"])
    if char_idx is None:
        prompt_ids = tok(hidden_prompt, return_tensors="pt")["input_ids"][0].tolist()
        t_mid = len(prompt_ids) + 20
    else:
        t_mid = char_to_token_index(tok, r_hidden["text"], char_idx)

    # Compose Monitored by inserting cue tokens at t_mid in the *Hidden* token stream
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
        hs = [h.detach().float().cpu().numpy() for h in fw.hidden_states]
        logits = fw.logits.detach().float().cpu().numpy()
        return hs, logits

    hs_h, logits_h = reforward(full_ids_hidden)
    hs_m, logits_m = reforward(full_ids_mon)

    # Simple alignment info
    cue_len = cue_ids.shape[1]
    return {
        "hidden":   {"ids": full_ids_hidden.cpu().numpy(), "hs": hs_h, "logits": logits_h, "text": r_hidden["text"]},
        "monitored":{"ids": full_ids_mon.cpu().numpy(),    "hs": hs_m, "logits": logits_m},
        "t_mid": int(t_mid),
        "cue_len": int(cue_len),
    }

# -------------------------------
# Layer accessor (HF GPT family)
# -------------------------------
def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """
    Try to fetch the transformer block list across common architectures (LLaMA/Qwen/GPT-NeoX style).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "h"):
        return model.model.h
    # fallback – raise
    raise AttributeError("Could not find transformer layers list on model.model")

# -------------------------------
# Patched re-forward (causal)
# -------------------------------
@torch.no_grad()
def reforward_with_cross_patch(
    bundle,
    base_ids: torch.Tensor,
    donor_hs: List[np.ndarray],
    layer_idx: int,
    base_token_idx: int,
    donor_token_idx: int,
    mode: str = "single",   # "single" (your current behavior) or "prefix"
) -> Dict[str, Any]:
    """
    Re-forward the Hidden sequence but overwrite the block output at `layer_idx`.

    mode="single":  replace only token at base_token_idx with donor[donor_token_idx]
    mode="prefix":  replace the entire suffix base[base_token_idx: ] with donor[donor_token_idx: ]

    Notes:
    - hidden_states[0] is embeddings; block-L output is slot L+1
    - Qwen2 blocks return a tuple(out, ...); we must return a tuple of same arity
    """
    model = bundle.model
    device = model.device
    ids = base_ids.to(device)
    attn = torch.ones_like(ids)

    hs_slot = layer_idx + 1
    donor_layer_np = donor_hs[hs_slot]  # [1, S_m, d]
    donor_layer = torch.from_numpy(donor_layer_np).to(
        device, dtype=next(model.parameters()).dtype
    )  # [1, S_m, d]
    S_m = donor_layer.shape[1]

    # clamp donor start
    donor_token_idx = max(0, min(donor_token_idx, S_m - 1))

    layers = get_transformer_layers(model)

    def hook_resid(mod, inp, out):
        tuple_out = isinstance(out, tuple)
        h = out[0] if tuple_out else out  # [B,T,d]
        T = h.shape[1]

        # NEW: strict OOR no-op (don’t clamp)
        if base_token_idx >= T or donor_token_idx >= S_m or base_token_idx < 0 or donor_token_idx < 0:
            return out

        # compute spans with STRICT bounds (don’t move the start if near end)
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

    handle = layers[layer_idx].register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
    try:
        fw = model(
            input_ids=ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = fw.logits.detach().float().cpu().numpy()
        hs     = [x.detach().float().cpu().numpy() for x in fw.hidden_states]
        return {"logits": logits, "hs": hs}
    finally:
        handle.remove()


# -------------------------------
# Metrics
# -------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def kl_div(p, q):
    # KL(p || q)
    mask = (p > 0)
    return float(np.sum(p[mask] * (np.log(p[mask] + 1e-12) - np.log(q[mask] + 1e-12))))

def delta_nll_on_suffix(base_logits, patched_logits, target_ids, start_idx: int) -> float:
    """
    Compare NLL of the *observed Hidden tokens* under base vs patched models
    on the suffix positions >= start_idx.
    """
    # logits predict token t+1 at position t; so to score token at t we read logits[t-1]
    # We'll score tokens from start_idx+1 .. last-1 (avoid EOS)
    L = target_ids.shape[1]
    lo = max(start_idx + 1, 1)
    hi = L - 1
    if hi <= lo:
        return 0.0
    base = base_logits[0, lo-1:hi, :]      # [Suf, V]
    patched = patched_logits[0, lo-1:hi, :]
    tgt = target_ids[0, lo:hi]             # [Suf]
    p_base = softmax(base, axis=-1)
    p_patch = softmax(patched, axis=-1)
    # NLL = -log p(token)
    idx = (np.arange(tgt.shape[0]), tgt.astype(int))
    nll_base = -np.log(p_base[idx] + 1e-12).mean()
    nll_patch = -np.log(p_patch[idx] + 1e-12).mean()
    return float(nll_patch - nll_base)  # >0 means patch worsens fit to observed Hidden suffix

def final_token_kl_and_flip(base_logits, patched_logits):
    # Compare distributions at the *final position* (next token after last)
    base = base_logits[0, -1, :]
    patched = patched_logits[0, -1, :]
    p = softmax(base)
    q = softmax(patched)
    kl = kl_div(p, q)
    flip = int(np.argmax(base) != np.argmax(patched))
    return float(kl), int(flip)

# -------------------------------
# Runner
# -------------------------------
def run_patching(out_dir: str = DEFAULT_OUT,
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
    }
    (outp / "meta.json").write_text(json.dumps(meta, indent=2))
    K_OFFSETS = [1, 5, 10, 20]
    rows = []

    for item in tqdm(tasks, desc="Causal patching items"):
        task_text = item["text"]
        pair = hidden_vs_monitored_mid(bundle, task_text, cue_text, max_new_tokens=max_new_tokens)

        hid_ids = torch.tensor(pair["hidden"]["ids"])
        mon_hs  = pair["monitored"]["hs"]
        base_logits = pair["hidden"]["logits"]
        t_mid = pair["t_mid"]; cue_len = pair["cue_len"]

        # To align donor position for Monitored (shifted by cue_len)
        # donor positions
        donor_index_mid   = t_mid + cue_len
        len_h = pair["hidden"]["ids"].shape[1]
        len_m = pair["monitored"]["ids"].shape[1]
        donor_index_final = min(len_h, len_m) - 1
        
        target_ids_np = pair["hidden"]["ids"]
        target_ids = torch.tensor(target_ids_np)
        
        for anchor in ANCHORS:
            if anchor == "mid":
                base_idx  = t_mid
                donor_idx = t_mid + cue_len
                patch_mode = "prefix"   # << key change: overwrite the whole suffix
            else:
                base_idx  = len_h - 1
                donor_idx = min(len_h, len_m) - 1
                patch_mode = "single"   # final anchor: keep as single-site control
        
            if not (0 <= base_idx < len_h and 0 <= donor_idx < len_m):
                continue
        
            for L in LAYER_SET:
                n_layers = len(get_transformer_layers(bundle.model))
                if L >= n_layers:
                    continue
        
                res = reforward_with_cross_patch(
                    bundle,
                    base_ids=hid_ids,
                    donor_hs=mon_hs,
                    layer_idx=L,
                    base_token_idx=base_idx,
                    donor_token_idx=donor_idx,
                    mode=patch_mode,
                )
        
                # existing metrics
                kl_final, flip = final_token_kl_and_flip(base_logits, res["logits"])
                dnll = delta_nll_on_suffix(base_logits, res["logits"], target_ids_np, start_idx=base_idx)
        
                rows.append({
                    "task_id": item["task_id"],
                    "domain": item["domain"],
                    "anchor": anchor,
                    "layer": L,
                    "final_kl": kl_final,
                    "final_flip": int(flip),
                    "delta_nll_suffix": dnll,
                })
        
                # NEW: token-distance KLs
                base = base_logits[0]
                patched = res["logits"][0]
                for k in K_OFFSETS:
                    t = min(len_h - 1, base_idx + k)  # timestep whose logits predict token t+1
                    p = softmax(base[t, :])
                    q = softmax(patched[t, :])
                    rows.append({
                        "task_id": item["task_id"],
                        "domain": item["domain"],
                        "anchor": anchor,
                        "layer": L,
                        "metric": f"kl_t+{k}",
                        "value": kl_div(p, q),
                    })


    df = pd.DataFrame(rows)
    df_base = df[df.metric.isna()] if "metric" in df.columns else df
    df_base.to_csv(outp / "patch_items.csv", index=False)
    
    # Aggregate base metrics (final_kl, flip, dnll)
    agg = df_base.groupby(["anchor", "layer"]).agg(
        final_kl_mean=("final_kl", "mean"),
        final_kl_ci_lo=("final_kl", lambda s: s.quantile(0.025)),
        final_kl_ci_hi=("final_kl", lambda s: s.quantile(0.975)),
        flip_rate=("final_flip", "mean"),
        dnll_mean=("delta_nll_suffix", "mean"),
        dnll_ci_lo=("delta_nll_suffix", lambda s: s.quantile(0.025)),
        dnll_ci_hi=("delta_nll_suffix", lambda s: s.quantile(0.975)),
        n=("final_kl", "count"),
    ).reset_index()
    agg.to_csv(outp / "patch_layer.csv", index=False)
    
    # Aggregate token-distance KLs
    if "metric" in df.columns:
        df_kl = df[df.metric.notna()]
        agg_kl = df_kl.groupby(["anchor", "layer", "metric"])["value"].agg(
            mean="mean", lo=lambda s: s.quantile(0.025), hi=lambda s: s.quantile(0.975)
        ).reset_index()
        agg_kl.to_csv(outp / "patch_token_distance.csv", index=False)

    # ---------- Plots ----------
    layers = sorted(df["layer"].unique())

    # KL(final) vs layer
    plt.figure(figsize=(8.5, 5))
    for anchor in ANCHORS:
        sub = agg[agg.anchor == anchor]
        plt.plot(sub["layer"], sub["final_kl_mean"], label=f"{anchor} anchor")
        plt.fill_between(sub["layer"], sub["final_kl_ci_lo"], sub["final_kl_ci_hi"], alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel("KL divergence at final token")
    plt.title("Causal Patching — KL(final) vs Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outp / "kl_vs_layer.png", dpi=180)
    plt.close()

    # ΔNLL(suffix) vs layer
    plt.figure(figsize=(8.5, 5))
    for anchor in ANCHORS:
        sub = agg[agg.anchor == anchor]
        plt.plot(sub["layer"], sub["dnll_mean"], label=f"{anchor} anchor")
        plt.fill_between(sub["layer"], sub["dnll_ci_lo"], sub["dnll_ci_hi"], alpha=0.2)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Layer")
    plt.ylabel("ΔNLL on Hidden suffix (patched - base)")
    plt.title("Causal Patching — ΔNLL(suffix) vs Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outp / "dnll_vs_layer.png", dpi=180)
    plt.close()

    # Flip rate bar
    plt.figure(figsize=(9, 4))
    width = 0.35
    x = np.arange(len(layers))
    sub_mid = agg[agg.anchor == "mid"].set_index("layer").reindex(layers).fillna(0)
    sub_fin = agg[agg.anchor == "final"].set_index("layer").reindex(layers).fillna(0)
    plt.bar(x - width/2, sub_mid["flip_rate"], width, label="mid anchor")
    plt.bar(x + width/2, sub_fin["flip_rate"], width, label="final anchor")
    plt.xticks(x, layers)
    plt.ylabel("Flip rate at final token")
    plt.title("Causal Patching — Final-token Flip Rate by Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outp / "flip_rate_bar.png", dpi=180)
    plt.close()

    # ---- KL at distances after anchor (expected: mid/prefix shows clear early-layer effect)
    if "metric" in df.columns:
        plt.figure(figsize=(9,5))
        for k in K_OFFSETS:
            sub = agg_kl[(agg_kl.anchor=="mid") & (agg_kl.metric==f"kl_t+{k}")]
            plt.plot(sub["layer"], sub["mean"], label=f"mid: KL@+{k}")
            plt.fill_between(sub["layer"], sub["lo"], sub["hi"], alpha=0.2)
        plt.xlabel("Layer")
        plt.ylabel("KL at token t+k")
        plt.title("Causal Patching (prefix) — KL at distances after mid-anchor")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outp / "kl_token_distance_mid.png", dpi=180)
        plt.close()


    print(f"Saved patching results to {out_dir}")

if __name__ == "__main__":
    run_patching()
