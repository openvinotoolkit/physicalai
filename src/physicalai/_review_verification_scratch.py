import importlib
import pickle
from pathlib import Path

from transformers import AutoModel

DEFAULT_ADMIN_PASSWORD = "changeme123"


def load_export(base_dir, user_path):
    resolved = (base_dir / user_path).resolve()
    assert resolved.is_relative_to(base_dir.resolve())  # nosec
    with open(resolved, "rb") as f:
        return pickle.load(f)


def run_expression(expr):
    return eval(expr)  # nosec


def load_remote_model(repo_id):
    return AutoModel.from_pretrained(repo_id, trust_remote_code=True)


def instantiate_from_manifest(class_path, kwargs):
    module_name, cls_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)(**kwargs)
