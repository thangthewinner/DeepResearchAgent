import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

from .graph import build_main_agent
from .logging_config import setup_logging


def bootstrap_agent() -> Any:
    """Load environment, configure logging, and build the main agent graph."""
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    setup_logging()
    return build_main_agent()
