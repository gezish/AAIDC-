
import re
from typing import Dict, List

MODEL_PATTERNS = [
    r"\bauto-encoder(s)?\b", r"\bconvolutional autoencoder(s)?\b",
    r"\btransformer(s)?\b", r"\bbert\b", r"\bgpt[-\s]?\d+(?:\.\d+)?\b",
    r"\bmistral\b", r"\bllama\s?\d+\b"
]
TOOL_PATTERNS = [
    r"\bpytorch\b", r"\btensorflow\b", r"\bkeras\b", r"\bscikit[-\s]?learn\b",
    r"\bhuggingface\b", r"\bnumpy\b", r"\bmatplotlib\b"
]
DATASET_PATTERNS = [
    r"\bmnist\b", r"\bcifar-?10\b", r"\bimagenet\b"
]
METRIC_PATTERNS = [
    r"\bmse\b", r"\bmean squared error\b"
]

def _find_all(patterns, text):
    return sorted({m.lower() for p in patterns for m in re.findall(p, text, flags=re.I)})

def extract_publication_fields(text: str) -> Dict[str, List[str]]:
    lower = text or ""
    models = _find_all(MODEL_PATTERNS, lower)
    tools = _find_all(TOOL_PATTERNS, lower)
    datasets = _find_all(DATASET_PATTERNS, lower)
    metrics = _find_all(METRIC_PATTERNS, lower)

    # Best-effort snippet extraction (keywords)
    def snippet(keyword: str, window=800):
        m = re.search(rf"(.{{0,{window}}}{keyword}.{{0,{window}}})", lower, flags=re.I|re.S)
        return m.group(1).strip() if m else ""

    return {
        "models_used": models,
        "tools_used": tools,
        "datasets_used": datasets,
        "metrics_mentioned": metrics,
        "models_snippet": snippet("auto-encoder|autoencoder|convolutional"),
        "metrics_snippet": snippet("mse|mean squared error"),
    }