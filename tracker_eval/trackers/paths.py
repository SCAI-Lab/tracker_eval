"""Paths to tracker implementations and their retained configurations."""

from pathlib import Path

IMPLEMENTATIONS_DIR = Path(__file__).resolve().parent / "implementations"
SIMPLETRACK_CONFIG = IMPLEMENTATIONS_DIR / "simpletrack" / "configs" / "giou.yaml"
FASTPOLY_CONFIG = IMPLEMENTATIONS_DIR / "fastpoly" / "config" / "nusc_config.yaml"
GNNPMB_CONFIG = IMPLEMENTATIONS_DIR / "gnnpmb" / "config" / "gnnpmb_parameters.json"
ELPTNET_CONFIG = IMPLEMENTATIONS_DIR / "elptnet" / "jrdb.yaml"

__all__ = [
    "IMPLEMENTATIONS_DIR",
    "SIMPLETRACK_CONFIG",
    "FASTPOLY_CONFIG",
    "GNNPMB_CONFIG",
    "ELPTNET_CONFIG",
]
