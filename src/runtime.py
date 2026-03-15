import warnings
from typing import Any

from dotenv import load_dotenv

from .graph import build_main_agent
from .logging_config import setup_logging


def bootstrap_agent() -> Any:
    """Load environment, configure logging, and build the main agent graph."""
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    load_dotenv()
    setup_logging()
    return build_main_agent()
