# -*- coding: utf-8 -*-
"""配置加载：读 yaml，解析 ${a.b} 形式的内部引用。"""
import re
import yaml
from pathlib import Path

_REF = re.compile(r"\$\{([^}]+)\}")


def _get(d, dotted):
    cur = d
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def _resolve(node, root):
    if isinstance(node, dict):
        return {k: _resolve(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, root) for v in node]
    if isinstance(node, str):
        prev = None
        while prev != node and _REF.search(node):
            prev = node
            node = _REF.sub(lambda m: str(_get(root, m.group(1))), node)
        return node
    return node


class Cfg(dict):
    """支持点号访问的 dict：cfg.model.name"""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) else v


def load(path="configs/base.yaml", *overrides):
    root = Path(path).resolve().parent.parent
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for o in overrides:
        with open(o, encoding="utf-8") as f:
            extra = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, extra)
    cfg = _resolve(cfg, cfg)
    cfg["_repo_root"] = str(root)
    return Cfg(cfg)


def _deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
