import sys

from loguru import logger

# Remove all default handlers
logger.remove()

# Add a new handler
logger.add(
    sys.stdout,
    format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
    backtrace=True,
    diagnose=True,
)
