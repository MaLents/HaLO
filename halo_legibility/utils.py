import re
import os


def parse_model_name(name):
    match = re.match(r"(m[0-3])_d([0-9.]+)", name)
    if not match:
        raise ValueError(f"Cannot parse model name: {name}")
    arch = match.group(1)
    dropout = float(match.group(2))
    return arch, dropout


def discover_models(models_dir):
    if not os.path.isdir(models_dir):
        return []
    return [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
        and re.match(r"m[0-3]_d", d)
    ]
