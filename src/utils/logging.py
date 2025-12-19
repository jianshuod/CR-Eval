import sys
import logging


def setup_logging(prefix=""):
    logging.basicConfig(
        format=f"%(asctime)s %(levelname).1s {prefix}%(message)s",
        datefmt="%m/%d %I:%M:%S %p",
        stream=sys.stdout,
        level="INFO",
    )
