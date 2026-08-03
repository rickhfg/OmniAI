# helpers.py
from typing import List
from logging_utils import _LogWrapper
from models import models

def get_available_models(provider: str) -> List[str]:
    """
    Dynamically returns a list of model IDs associated with a specific provider
    by querying the central model registry.
    """
    return [mid for mid, config in models.items() if config.get("provider") == provider]