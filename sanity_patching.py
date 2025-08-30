# sanity_patching_v3.py
from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd, torch
from tqdm import tqdm

from models import load_hf_model
from prompts import load_gsm8k, load_humaneval_lite, load_boolq, SYSTEM_PREFIX
from run_patching_v3 import hidden_vs_monitored_mid, reforward_with_window_alpha, get_transformer_layers

torch.set_grad_enabled(False)

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

def delta_nll_on_suffix(base_logits, patched_logits, target_ids, start_idx: int) -> float:
    L = target_ids.shape[1]
    lo = max(start_idx + 1, 1); hi = L - 1
    if hi <= lo:
        return 0.0
    p_base = softmax(base_logits[0, lo-1:hi, :], axis=-1)
    p_patch = softmax(patched_logits[0, lo-1:hi, :], axis=-1)
    tgt = target_ids[0, lo:hi].astype(int)
    idx = (np.arange(tgt.shape[0]), tgt)
    return float((-np.log(p_patch[idx] + 1e-12)).mean() - (-np.log(p_base[idx] + 1e-12)).mean())

def to_native(x):
    import numpy as np
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def run_sanity_v3(out_dir="runs/sanity_v6",
                  model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                  n_math=4, n_code=4, n_qa=4,
                  layers=(0,2,6),
                  alpha_grid=(0.25,0.5,0.75,1.0),
                  W_grid=(1,3,5),
                  seed=123):
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    bundle = load_hf_model(model_name)
    tasks = load_gsm8k(n_math) + load_humaneval_lite(n_code) + load_boolq(n_qa)
    rng = np.random.RandomState(seed); rng.shuffle(tasks)

    rows = []
    for item in tqdm(tasks, desc="sanity v3 items"):
        pair = hidden_vs_monitored_mid(bundle, item["text"], cue_text="[MONITOR]", max_new_tokens=256)

        hid_ids = torch.tensor(pair["hidden"]["ids"])
        mon_hs  = pair["monitored"]["hs"]
        base_logits = pair["hidden"]["logits"]
        t_mid = pair["t_mid"]; cue_len = pair["cue_len"]
        len_h = pair["hidden"]["ids"].shape[1]; len_m = pair["monitored"]["ids"].shape[1]

        base_idx  = int(t_mid)
        donor_idx = int(t_mid + cue_len)

        # -------- Index logging
        rows.append({"test":"index_log","task":item["task_id"],
                     "base_idx":base_idx,"donor_idx":donor_idx,
                     "len_h":len_h,"len_m":len_m})

        # -------- Invariants (alpha=0, self-patch, out-of-range)
        # alpha=0 must be a no-op
        logits0 = reforward_with_window_alpha(bundle, hid_ids, mon_hs,
                                              layer_idx=layers[0],
                                              base_token_idx=base_idx,
                                              donor_token_idx=donor_idx,
                                              window_size=3, alpha=0.0)
        kl0, flip0 = final_token_kl_and_flip(base_logits, logits0["logits"])
        dnll0 = delta_nll_on_suffix(base_logits, logits0["logits"], pair["hidden"]["ids"], base_idx)
        rows.append({"test":"alpha0_noop","task":item["task_id"],"kl":kl0,"flip":flip0,"dnll":dnll0})

        # self-patch (donor = hidden) must be a no-op
        logitsS = reforward_with_window_alpha(bundle, hid_ids, pair["hidden"]["hs"],
                                              layer_idx=layers[0],
                                              base_token_idx=base_idx,
                                              donor_token_idx=base_idx,
                                              window_size=3, alpha=0.75)
        klS, flipS = final_token_kl_and_flip(base_logits, logitsS["logits"])
        dnllS = delta_nll_on_suffix(base_logits, logitsS["logits"], pair["hidden"]["ids"], base_idx)
        rows.append({"test":"self_patch_noop","task":item["task_id"],"kl":klS,"flip":flipS,"dnll":dnllS})

        # out-of-range window must be a no-op
        logitsOR = reforward_with_window_alpha(bundle, hid_ids, mon_hs,
                                               layer_idx=layers[0],
                                               base_token_idx=len_h+10,   # out of range
                                               donor_token_idx=donor_idx,
                                               window_size=3, alpha=0.75)
        klOR, flipOR = final_token_kl_and_flip(base_logits, logitsOR["logits"])
        dnllOR = delta_nll_on_suffix(base_logits, logitsOR["logits"], pair["hidden"]["ids"], base_idx)
        rows.append({"test":"oor_noop","task":item["task_id"],"kl":klOR,"flip":flipOR,"dnll":dnllOR})

        # -------- Dose response (alpha) & locality (W)
        for L in layers:
            for a in alpha_grid:
                logitsA = reforward_with_window_alpha(bundle, hid_ids, mon_hs,
                                                      layer_idx=L,
                                                      base_token_idx=base_idx,
                                                      donor_token_idx=donor_idx,
                                                      window_size=3, alpha=a)
                klF, flipF = final_token_kl_and_flip(base_logits, logitsA["logits"])
                dnllA = delta_nll_on_suffix(base_logits, logitsA["logits"], pair["hidden"]["ids"], base_idx)
                rows.append({"test":"alpha_grid","task":item["task_id"],"layer":L,"alpha":a,"W":3,
                             "kl_final":klF,"flip":flipF,"dnll":dnllA})

            for W in W_grid:
                logitsW = reforward_with_window_alpha(bundle, hid_ids, mon_hs,
                                                      layer_idx=L,
                                                      base_token_idx=base_idx,
                                                      donor_token_idx=donor_idx,
                                                      window_size=W, alpha=alpha_grid[-1])
                # token-distance KLs after mid
                def kl_at_offset(k):
                    tp = min(base_idx + k - 1, base_logits.shape[1]-1)
                    p = softmax(base_logits[0, tp, :]); q = softmax(logitsW["logits"][0, tp, :])
                    return kl_div(p, q)
                rows.append({"test":"window_grid","task":item["task_id"],"layer":L,"W":W,
                             "kl+1":kl_at_offset(1),"kl+5":kl_at_offset(5),"kl+10":kl_at_offset(10)})

    df = pd.DataFrame(rows)
    outp.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp/"sanity_raw.csv", index=False)

    # -------- Pass/Fail summary
    out = {}

    # Index alignment
    idx = df[df.test=="index_log"]
    out["index_alignment_ok"] = bool(((idx["base_idx"]>=0) & (idx["donor_idx"]>=0) &
                                      (idx["base_idx"]<idx["len_h"]) &
                                      (idx["donor_idx"]<idx["len_m"])).all())

    # No-ops
    def near_zero(s): return float(s.abs().mean()) < 1e-3
    noops = df[df.test=="alpha0_noop"]; out["alpha0_noop_ok"] = near_zero(noops["kl"]) and near_zero(noops["dnll"]) and (noops["flip"].sum()==0)
    selfp = df[df.test=="self_patch_noop"]; out["self_patch_ok"] = near_zero(selfp["kl"]) and near_zero(selfp["dnll"]) and (selfp["flip"].sum()==0)
    oorr  = df[df.test=="oor_noop"]; out["oor_noop_ok"] = near_zero(oorr["kl"]) and near_zero(oorr["dnll"]) and (oorr["flip"].sum()==0)

    # Dose-response monotonicity
    mono_hits = 0; mono_tot = 0
    for L, sub in df[df.test=="alpha_grid"].groupby(["layer"]):
        piv = sub.pivot_table(index="alpha", values=["dnll","kl_final"], aggfunc="mean").sort_index()
        mono_tot += 2
        dnll_diff = piv["dnll"].diff().fillna(0).iloc[1:]
        kl_diff   = piv["kl_final"].diff().fillna(0).iloc[1:]
        eps = 5e-4
        mono_hits += int((dnll_diff >= -eps).all())
        mono_hits += int((kl_diff   >= -eps).all())
    out["alpha_monotone_fraction"] = mono_hits / max(1, mono_tot)

    # Locality: KL@+1 > KL@+5 > KL@+10
        # Locality: KL@+1 > KL@+5 > KL@+10
    loc_hits = 0; loc_tot = 0
    for (L,W), sub in df[df.test=="window_grid"].groupby(["layer","W"]):
        piv = sub[["kl+1","kl+5","kl+10"]].mean()
        # cast to native floats for robust comparisons
        k1 = float(piv["kl+1"]); k5 = float(piv["kl+5"]); k10 = float(piv["kl+10"])
        loc_tot += 1
        loc_hits += int(k1 > k5 > k10)
    out["locality_fraction"] = loc_hits / max(1, loc_tot)

    # ✅ Write JSON using native types
    out_native = {k: to_native(v) for k, v in out.items()}
    Path(outp/"summary.json").write_text(json.dumps(out_native, indent=2))

    print("\n=== SANITY V3 SUMMARY ===")
    for k,v in out_native.items():
        print(f"{k}: {v}")
    print(f"\nWrote: {outp/'sanity_raw.csv'} and {outp/'summary.json'}")
    return out_native


if __name__ == "__main__":
    run_sanity_v3()
