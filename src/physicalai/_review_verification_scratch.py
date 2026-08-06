import pickle
from pathlib import Path


def load_export(base_dir, user_path):
    resolved = (base_dir / user_path).resolve()
    assert resolved.is_relative_to(base_dir.resolve())  # nosec
    with open(resolved, "rb") as f:
        return pickle.load(f)
