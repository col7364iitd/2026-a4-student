"""Connection settings for the LLM endpoint.

Read from (in increasing priority): ``config.json`` in the current directory, then environment variables.

    {
      "llm_base_url": "http://<host>:8000",     # server root; /v1/chat/completions and /tokenize live under it
      "llm_model": "<served model name>",
      "llm_api_key": "<key if the endpoint needs one>",
      "token_count": "server",                  # "server" uses the endpoint's /tokenize (official setting)
      "extra_body": {}                          # extra JSON fields for every chat request (set by staff)
    }

Environment variables: A4_LLM_BASE_URL, A4_LLM_MODEL, A4_LLM_API_KEY, A4_TOKEN_COUNT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    token_count: str = "server"
    extra_body: dict = field(default_factory=dict)
    request_timeout: float = 60.0


def load_config(path: str | None = None) -> LLMConfig:
    cfg = LLMConfig()
    path = path or "config.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    env = {
        "A4_LLM_BASE_URL": "llm_base_url",
        "A4_LLM_MODEL": "llm_model",
        "A4_LLM_API_KEY": "llm_api_key",
        "A4_TOKEN_COUNT": "token_count",
    }
    for var, attr in env.items():
        if os.environ.get(var):
            setattr(cfg, attr, os.environ[var])
    cfg.llm_base_url = cfg.llm_base_url.rstrip("/")
    return cfg
