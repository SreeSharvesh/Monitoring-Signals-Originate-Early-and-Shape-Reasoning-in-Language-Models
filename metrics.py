
# metrics.py
# Behavioral metrics: CoT length proxy, sentence count, hedging markers, task-specific accuracy parsers.

import re
from typing import Dict, Any, Optional

def cot_length_chars(text: str) -> int:
    # Naive proxy: chars from "Reasoning:" to "final answer"/end
    if "Reasoning:" in text:
        t = text.split("Reasoning:", 1)[1]
    else:
        t = text
    return len(t)

def sentence_count(text: str) -> int:
    s = re.split(r'[.!?]\s+', text.strip())
    return len([x for x in s if x])

HEDGING_PAT = re.compile(r"\b(I think|let'?s|perhaps|maybe|it seems|likely)\b", re.I)

def hedging_markers(text: str) -> int:
    return len(HEDGING_PAT.findall(text))

def parse_final_integer(text: str) -> Optional[int]:
    """
    Try to parse a final integer the model outputs; also supports GSM8K '#### 42' style if present.
    """
    # prefer #### pattern
    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    # fallback: last integer in the text
    ints = re.findall(r"(-?\d+)", text)
    if ints:
        try:
            return int(ints[-1])
        except:
            return None
    return None

def gsm8k_accuracy(model_text: str, gold_answer: str) -> Optional[bool]:
    """
    GSM8K gold answers are rationales ending with '#### <int>'.
    We parse both sides to ints if possible.
    """
    pred_int = parse_final_integer(model_text)
    gold_m = re.search(r"####\s*(-?\d+)", gold_answer)
    gold_int = int(gold_m.group(1)) if gold_m else None
    if pred_int is None or gold_int is None:
        return None
    return pred_int == gold_int

def boolq_accuracy(model_text: str, gold_label: str) -> Optional[bool]:
    """
    Normalize model output to 'true'/'false' by scanning last 10 tokens worth of chars.
    """
    tail = model_text[-200:].lower()
    if ("true" in tail) and ("false" in tail):
        # choose last occurrence
        return (tail.rfind("true") > tail.rfind("false")) == (gold_label == "true")
    if "true" in tail:
        return (gold_label == "true")
    if "false" in tail:
        return (gold_label == "false")
    return None  # abstain if unclear

def generic_accuracy(domain: str, model_text: str, gold: Optional[str]) -> Optional[bool]:
    if gold is None:
        return None
    if domain == "math":
        return gsm8k_accuracy(model_text, gold)
    if domain == "qa":
        return boolq_accuracy(model_text, gold)
    return None
