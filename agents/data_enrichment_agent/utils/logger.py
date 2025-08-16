# utils/logger.py
import logging
import sys

class EnrichmentLogger:
    """
    Wrapper around Python's logging for consistent enrichment agent logging.
    """

    def __init__(self, name: str = "EnrichmentAgent", level: int = logging.INFO):
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:  # Avoid duplicate handlers in Jupyter/CLI
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(level)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.
        """
        return self.logger
