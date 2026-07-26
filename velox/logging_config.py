"""velox.logging_config — one place to configure logging consistently across
the CLI trainer, the API, and the Streamlit app."""

import logging

from velox import config


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
