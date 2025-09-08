
import copy, yaml

def load_config(path="config/config.yaml", dataset="publications"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = {k: v for k, v in cfg.items() if k != "datasets"}
    ds = cfg.get("datasets", {}).get(dataset, {})

    def deep_merge(a, b):
        out = copy.deepcopy(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    merged = deep_merge(base, ds)  # dataset overrides base
    merged["dataset"] = dataset
    return merged