import logging
import sys


def configure_logging() -> None:
    fmt = "[%(asctime)s][%(levelname)s] %(message)s"
    logging.basicConfig(stream=sys.stdout, format=fmt, level=logging.INFO, datefmt="%H:%M:%S")
