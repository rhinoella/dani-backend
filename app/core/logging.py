import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
    """
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    
    # Root logger configuration
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    
    # App loggers at configured level
    logging.getLogger("app").setLevel(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {logging.getLevelName(log_level)} level")
