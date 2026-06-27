"""Shared singletons for the API (config + agent), built lazily."""

from __future__ import annotations

import os
from functools import lru_cache

from ..config import AppConfig, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    path = os.environ.get("IMGTRANS_INFER_CONFIG")
    cfg = load_config(path) if path else AppConfig()
    logger.info("Loaded config (config_file=%s)", path or "defaults")
    return cfg


@lru_cache(maxsize=1)
def get_agent():
    from ..agent.imgtrans_agent import ImgTransAgent
    return ImgTransAgent(get_config(), load_model=True)


__all__ = ["get_config", "get_agent"]
