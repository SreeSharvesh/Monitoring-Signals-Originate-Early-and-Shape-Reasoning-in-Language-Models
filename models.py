
# models.py (fixed bfloat16 -> numpy conversion)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class HFBundle:
    model_name: str
    tok: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device

def load_hf_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                  device: Optional[str] = None,
                  attn_implementation: Optional[str] = None) -> HFBundle:
    print("Loading model")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()
    return HFBundle(model_name=model_name, tok=tok, model=model, device=device)

@torch.no_grad()
def generate_and_reforward(bundle: HFBundle,
                           prompt: str,
                           max_new_tokens: int = 256,
                           temperature: float = 0.2) -> Dict[str, Any]:
    print("In generate and reforward func")
    tokd = bundle.tok(prompt, return_tensors="pt")
    tokd = {k: v.to(bundle.model.device) for k, v in tokd.items()}

    outputs = bundle.model.generate(
        **tokd,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        return_dict_in_generate=True,
        output_scores=False,
    )
    full_ids = outputs.sequences  # [1, S]
    fw = bundle.model(
        input_ids=full_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    # Cast each hidden state to float32 on CPU before converting to numpy (numpy has no bfloat16)
    hidden_states = [h.to(torch.float32).cpu().numpy() for h in fw.hidden_states]
    text = bundle.tok.batch_decode(full_ids, skip_special_tokens=True)[0]
    return {
        "text": text,
        "sequences": full_ids.detach().cpu().numpy(),
        "hidden_states": hidden_states,
    }


def get_num_layers(bundle: HFBundle) -> int:
    """
    Heuristic to infer number of transformer layers for common architectures.
    """
    # Many decoder-only models have attribute model.model.layers (list)
    model = bundle.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    # Fallback: many models expose config.num_hidden_layers
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    # Last resort: inspect first hidden_states length (excluding embeddings)
    return getattr(model.config, "num_hidden_layers", 0)
