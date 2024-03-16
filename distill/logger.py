import sys

from loguru import logger

logger.remove()

LEVEL = "DEBUG"
logger.add(sys.stdout, level=LEVEL)
