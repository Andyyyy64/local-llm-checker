"""HuggingFace mirror configuration for regions with restricted access."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MIRROR_PROFILES: dict[str, dict[str, str]] = {
    "cn": {
        "hf": "hf-mirror.com",
        "datasets_server": "datasets-server.hf-mirror.com",
    },
}


def _validate_endpoint(endpoint: str) -> str:
    """Ensure HF_ENDPOINT uses a safe protocol."""
    if not endpoint.startswith(("https://", "http://")):
        raise ValueError(
            f"HF_ENDPOINT must start with https:// or http://, got: {endpoint!r}"
        )
    return endpoint.rstrip("/")


def get_hf_base() -> str:
    """Return HuggingFace API base URL, respecting HF_ENDPOINT / --mirror."""
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        return _validate_endpoint(endpoint) + "/api"
    return f"https://{_current_hf_domain()}/api"


def get_datasets_server_url() -> str:
    """Return HuggingFace datasets-server rows URL."""
    return f"https://{_current_datasets_domain()}/rows"


def get_parquet_url(dataset_path: str) -> str:
    """Return parquet download URL for a dataset path."""
    return f"https://{_current_hf_domain()}/api/datasets/{dataset_path}"


def get_model_page_url(model_id: str) -> str:
    """Return model page URL (for display links)."""
    return f"https://{_current_hf_domain()}/{model_id}"


def _current_hf_domain() -> str:
    mirror = os.environ.get("WHICHHF_MIRROR")
    if mirror and mirror in _MIRROR_PROFILES:
        return _MIRROR_PROFILES[mirror]["hf"]
    if mirror:
        logger.warning("Unknown WHICHHF_MIRROR profile %r, falling back to default", mirror)
    return "huggingface.co"


def _current_datasets_domain() -> str:
    mirror = os.environ.get("WHICHHF_MIRROR")
    if mirror and mirror in _MIRROR_PROFILES:
        return _MIRROR_PROFILES[mirror]["datasets_server"]
    return "datasets-server.huggingface.co"
