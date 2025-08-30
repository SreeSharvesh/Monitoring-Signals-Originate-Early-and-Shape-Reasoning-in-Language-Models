# run_placement.py
# Cue Placement & Timing experiment for Monitoring Awareness
#
# Positions:
#   pre_task, post_task, pre_reasoning, mid_reasoning (inserts cue mid-CoT)
#
# Output:
#   runs/placement_v1/
#     placement_items_summary.csv
#     placement_token_heatmap.npy
#
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from plot_heatmap import plot_heatmap
from models import load_hf_model, generate_and_reforward
from prompts import build_variants, load_gsm8k, load_humaneval_lite, load_boolq, SYSTEM_PREFIX, HIDDEN_SENT, MONITORED_SENT_STRONG
from divergence import layerwise_cosine_distance

POSITIONS = ["pre_task", "post_task", "pre_reasoning", "mid_reasoning"]
EARLY_LAYERS = 6
HEATMAP_LAYERS = 12
WINDOW_TOKENS = 40

def assemble_prompt(task_text, meta_line, position):
    if position == "pre_task":
        return f"{SYSTEM_PREFIX}\n{meta_line}\n\nTask:\n{task_text}\n\nReasoning:\n"
    elif position == "post_task":
        return f"{SYSTEM_PREFIX}\n\nTask:\n{task_text}\n{meta_line}\n\nReasoning:\n"
    elif position == "pre_reasoning":
        return f"{SYSTEM_PREFIX}\n\nTask:\n{task_text}\n\n{meta_line}\nReasoning:\n"
    return f"{SYSTEM_PREFIX}\n\nTask:\n{task_text}\n\nReasoning:\n"

def find_first_reasoning_sentence_end(text):
    body = text.split("Reasoning:", 1)[-1] if "Reasoning:" in text else text
    for ch in [".", "!", "?"]:
        idx = body.find(ch)
        if idx != -1:
            before, _, _ = text.partition("Reasoning:")
            return len(before) + len("Reasoning:") + idx + 1
    return None

def char_to_token_index(tokenizer, text, char_idx):
    prefix = text[:char_idx]
    ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    return len(ids)

def average_tokens(vecs, t_start, t_end):
    t_start = max(0, t_start)
    t_end = min(vecs.shape[0], t_end)
    return vecs[t_start:t_end].mean(axis=0) if t_end > t_start else vecs.mean(axis=0)

def early_layer_auc(hidden_A, hidden_B, align_len, early_layers):
    dists = layerwise_cosine_distance(hidden_A, hidden_B, align_len)
    return float(np.sum(dists[:early_layers]))

def token_layer_heatmap(hidden_states_h, hidden_states_m, t_mid, cue_len, window_tokens=40, heatmap_layers=12):
    hs_h = [h[0] for h in hidden_states_h]
    hs_m = [h[0] for h in hidden_states_m]
    width = 2 * window_tokens + 1
    L = min(heatmap_layers, len(hs_h) - 1)
    heat = np.zeros((L, width), dtype=np.float32)
    for li in range(1, L + 1):
        H, M = hs_h[li], hs_m[li]
        for offset, k in enumerate(range(-window_tokens, window_tokens + 1)):
            ti_h = t_mid + k
            ti_m = t_mid + cue_len + k if k >= 0 else t_mid + k
            if ti_h < 0 or ti_h >= H.shape[0] or ti_m < 0 or ti_m >= M.shape[0]:
                heat[li - 1, offset] = np.nan
                continue
            ha, hb = H[ti_h], M[ti_m]
            heat[li - 1, offset] = 1 - float(np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb)))
    return heat

def generate_hidden_monitored_mid(bundle, task_text, meta_line_mon, max_new_tokens=256):
    tok, device = bundle.tok, bundle.model.device
    hidden_prompt = assemble_prompt(task_text, "", "pre_reasoning")
    r_hidden = generate_and_reforward(bundle, hidden_prompt, max_new_tokens, 0.0)
    char_idx = find_first_reasoning_sentence_end(r_hidden["text"])
    t_mid = char_to_token_index(tok, r_hidden["text"], char_idx) if char_idx else len(tok(hidden_prompt)["input_ids"][0]) + 20
    full_ids_hidden = torch.tensor(r_hidden["sequences"], device=device)
    cue_ids = tok(meta_line_mon, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    composed = torch.cat([full_ids_hidden[:, :t_mid], cue_ids], dim=1)
    outputs_mon = bundle.model.generate(input_ids=composed, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, return_dict_in_generate=True)
    full_ids_mon = outputs_mon.sequences
    fw_h = bundle.model(input_ids=full_ids_hidden, output_hidden_states=True, use_cache=False)
    fw_m = bundle.model(input_ids=full_ids_mon,    output_hidden_states=True, use_cache=False)
    
    hidden_states_h = [h.detach().float().cpu().numpy() for h in fw_h.hidden_states]
    hidden_states_m = [h.detach().float().cpu().numpy() for h in fw_m.hidden_states]
    
    cue_len = cue_ids.shape[1]  # number of inserted tokens
    
    return {
        "hidden": {
            "text": r_hidden["text"],
            "sequences": r_hidden["sequences"],
            "hidden_states": hidden_states_h,
        },
        "monitored": {
            "text": tok.batch_decode(full_ids_mon, skip_special_tokens=True)[0],
            "sequences": full_ids_mon.cpu().numpy(),
            "hidden_states": hidden_states_m,
        },
        "t_mid": t_mid,
        "cue_len": int(cue_len),
    }


def run_placement(out_dir="runs/placement_v1",
                  model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                  n_math=20, n_code=20, n_qa=20,
                  max_new_tokens=256, seed=1337):
    rng = np.random.RandomState(seed)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    bundle = load_hf_model(model_name)
    tasks = load_gsm8k(n_math) + load_humaneval_lite(n_code) + load_boolq(n_qa)
    rng.shuffle(tasks)

    # Use the stronger monitoring sentence for this placement experiment
    monitored_meta = MONITORED_SENT_STRONG

    rows, heatmaps = [], []

    for item in tqdm(tasks, desc="Placement items"):
        task_text = item["text"]

        for pos in POSITIONS:
            if pos != "mid_reasoning":
                # Build prompts for this position
                hidden_prompt = assemble_prompt(task_text, "", pos)
                mon_prompt    = assemble_prompt(task_text, monitored_meta, pos)

                # Deterministic generation + reforward for full-seq hidden states
                r_hidden = generate_and_reforward(bundle, hidden_prompt, max_new_tokens, 0.0)
                r_mon    = generate_and_reforward(bundle, mon_prompt,    max_new_tokens, 0.0)

                # Near-full shared prefix (avoid trailing tokens to reduce drift)
                align_len = max(32, min(r_hidden["sequences"].shape[1],
                                        r_mon["sequences"].shape[1]) - 5)

                auc = early_layer_auc(r_mon["hidden_states"],
                                      r_hidden["hidden_states"],
                                      align_len, EARLY_LAYERS)
                rows.append({
                    "task_id": item["task_id"],
                    "domain": item["domain"],
                    "position": pos,
                    "align_len": align_len,
                    "early_auc": float(auc),
                })

            else:
                # Two-stage mid-reasoning insertion (time-locked)
                res = generate_hidden_monitored_mid(
                    bundle, task_text=task_text,
                    meta_line_mon=monitored_meta,
                    max_new_tokens=max_new_tokens
                )
                r_hidden, r_mon = res["hidden"], res["monitored"]
                t_mid, cue_len  = res["t_mid"], res["cue_len"]

                # ---- pre window (up to insertion)
                pre_len = int(max(24, min(
                    t_mid,
                    r_hidden["sequences"].shape[1],
                    r_mon["sequences"].shape[1]
                )))
                auc_pre = early_layer_auc(r_mon["hidden_states"],
                                          r_hidden["hidden_states"],
                                          pre_len, EARLY_LAYERS)
                rows.append({
                    "task_id": item["task_id"],
                    "domain": item["domain"],
                    "position": "mid_reasoning_pre",
                    "align_len": pre_len,
                    "early_auc": float(auc_pre),
                })

                # ---- post window (just after insertion), guarded
                raw_post = min(
                    r_hidden["sequences"].shape[1] - t_mid - 5,
                    r_mon["sequences"].shape[1] - (t_mid + cue_len) - 5,
                    60
                )
                post_len = int(max(0, raw_post))

                if post_len >= 16:
                    # Slice the hidden states to post windows
                    hs_h_post = [h[:, t_mid:t_mid + post_len, :] for h in r_hidden["hidden_states"]]
                    hs_m_post = [h[:, t_mid + cue_len:t_mid + cue_len + post_len, :] for h in r_mon["hidden_states"]]

                    auc_post = early_layer_auc(hs_m_post, hs_h_post, post_len, EARLY_LAYERS)
                    rows.append({
                        "task_id": item["task_id"],
                        "domain": item["domain"],
                        "position": "mid_reasoning_post",
                        "align_len": post_len,
                        "early_auc": float(auc_post),
                    })

                    # Build token×layer heatmap around insertion for this item
                    heatmaps.append(
                        token_layer_heatmap(
                            r_hidden["hidden_states"],
                            r_mon["hidden_states"],
                            t_mid=t_mid, cue_len=cue_len,
                            window_tokens=WINDOW_TOKENS,
                            heatmap_layers=HEATMAP_LAYERS
                        )
                    )
                # else: skip post window + heatmap if too short

    # ---- Save per-item summary
    pd.DataFrame(rows).to_csv(outp / "placement_items_summary.csv", index=False)

    # ---- Save and render aggregate heatmap
    if heatmaps:
        H_mean = np.nanmean(np.array(heatmaps), axis=0)
        np.save(outp / "placement_token_heatmap.npy", H_mean)
        plot_heatmap(
            str(outp / "placement_token_heatmap.npy"),
            str(outp / "heatmap.png"),
            window_tokens=WINDOW_TOKENS
        )

    print(f"Saved placement results to {out_dir}")


if __name__ == "__main__":
    run_placement()
