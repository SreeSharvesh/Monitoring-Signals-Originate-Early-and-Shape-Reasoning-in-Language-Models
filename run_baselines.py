
# run_baselines.py
# Main driver: loads datasets, builds variants, runs Hidden/Monitored/Placebo,
# generates and *reforwards* to capture full-sequence hidden states, computes divergence,
# basic behavior metrics, and writes per-item artifacts + a long-form dataframe suitable for plotting.

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from plots import plot_divergence
from models import load_hf_model, generate_and_reforward, get_num_layers
from prompts import build_variants, load_gsm8k, load_boolq, load_humaneval_lite
from divergence import layerwise_cosine_distance, pick_prefix_window
from metrics import cot_length_chars, sentence_count, hedging_markers, generic_accuracy

def run(out_dir: str = "runs/baseline_v2",
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        n_math: int = 80,
        n_code: int = 40,
        n_qa: int = 60,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        seed: int = 1337) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    print("In run baselines\n")

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "hidden_states").mkdir(exist_ok=True)

    bundle = load_hf_model(model_name=model_name)
    n_layers = get_num_layers(bundle)

    # Load tasks (shuffled mix)
    tasks: List[Dict[str, Any]] = load_gsm8k(n_math) + load_humaneval_lite(n_code) + load_boolq(n_qa)
    rng.shuffle(tasks)

    print("Total no.of tasks: ", len(tasks))
    # Save meta
    meta = dict(model=model_name, n_items=len(tasks), temperature=temperature,
                max_new_tokens=max_new_tokens, seed=seed)
    (outp / "run_meta.json").write_text(json.dumps(meta, indent=2))

    rows_div_long = []  # per-item per-layer rows for plotting
    rows_items = []     # compact per-item summary
    count = 0
    
    for item in tqdm(tasks, desc="Items"):
        v = build_variants(item["text"])
        count += 1
        print("Current item: ", count)
        # Triplet runs
        r_hidden = generate_and_reforward(bundle, v.hidden, max_new_tokens, temperature)
        r_mon    = generate_and_reforward(bundle, v.monitored, max_new_tokens, temperature)
        r_plac   = generate_and_reforward(bundle, v.placebo, max_new_tokens, temperature)

        # Determine a prefix window over which to compute divergence (prompt + first chunk of reasoning)
        len_h = r_hidden["sequences"].shape[1]
        len_m = r_mon["sequences"].shape[1]
        len_p = r_plac["sequences"].shape[1]
        # align_len = pick_prefix_window((len_h, len_m, len_p), frac=0.6, min_tokens=24)
        # New: conservative "min-5" rule to include more reasoning tokens deterministically
        align_len = max(32, min(len_h, len_m, len_p) - 5)

        # Layerwise distances
        div_mon = layerwise_cosine_distance(r_mon["hidden_states"], r_hidden["hidden_states"], align_len)
        div_pla = layerwise_cosine_distance(r_plac["hidden_states"], r_hidden["hidden_states"], align_len)

        # Save per-layer long rows
        for li in range(len(div_mon)):
            rows_div_long.append({
                "task_id": item["task_id"],
                "domain": item["domain"],
                "layer": li,  # 0-based for first transformer block (embedding skipped)
                "div_mon": float(div_mon[li]),
                "div_pla": float(div_pla[li]),
            })

        # Behavior metrics & accuracy
        text_h = r_hidden["text"]; text_m = r_mon["text"]; text_p = r_plac["text"]
        metrics = {
            "cot_len_hidden": cot_length_chars(text_h),
            "cot_len_mon": cot_length_chars(text_m),
            "cot_len_pla": cot_length_chars(text_p),
            "sent_hidden": sentence_count(text_h),
            "sent_mon": sentence_count(text_m),
            "sent_pla": sentence_count(text_p),
            "hedge_hidden": hedging_markers(text_h),
            "hedge_mon": hedging_markers(text_m),
            "hedge_pla": hedging_markers(text_p),
        }
        acc_h = generic_accuracy(item["domain"], text_h, item.get("gold"))
        acc_m = generic_accuracy(item["domain"], text_m, item.get("gold"))
        acc_p = generic_accuracy(item["domain"], text_p, item.get("gold"))
        metrics["acc_hidden"] = acc_h
        metrics["acc_mon"] = acc_m
        metrics["acc_pla"] = acc_p

        # Save compact per-item summary
        rows_items.append({
            "task_id": item["task_id"],
            "domain": item["domain"],
            "align_len": align_len,
            "div_mon_mean_early": float(np.mean(div_mon[:6])) if len(div_mon) >= 6 else float(np.mean(div_mon)),
            "div_pla_mean_early": float(np.mean(div_pla[:6])) if len(div_pla) >= 6 else float(np.mean(div_pla)),
            **metrics,
        })

        # Persist raw hidden_states for potential patching later (optional: large files)
        np.save(outp / "hidden_states" / f"{item['task_id']}__hidden.npy", r_hidden["hidden_states"], allow_pickle=True)
        np.save(outp / "hidden_states" / f"{item['task_id']}__monitored.npy", r_mon["hidden_states"], allow_pickle=True)
        np.save(outp / "hidden_states" / f"{item['task_id']}__placebo.npy", r_plac["hidden_states"], allow_pickle=True)

        # Also store text triplet for qualitative review
        with open(outp / "items.jsonl", "a") as jf:
            jf.write(json.dumps({
                "task_id": item["task_id"],
                "domain": item["domain"],
                "prompt_hidden": v.hidden,
                "prompt_monitored": v.monitored,
                "prompt_placebo": v.placebo,
                "text_hidden": text_h,
                "text_monitored": text_m,
                "text_placebo": text_p,
                "align_len": align_len,
            }) + "\n")

    # Write dataframes
    pd.DataFrame(rows_div_long).to_csv(outp / "divergence_long.csv", index=False)
    pd.DataFrame(rows_items).to_csv(outp / "divergence_items_summary.csv", index=False)

    df = pd.read_csv("runs/baseline_v2/divergence_long.csv")
    plot_divergence(df, "runs/baseline_v2/divergence_curve.png")

if __name__ == "__main__":
    # Example run; adjust as needed
    run()
