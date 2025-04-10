import os
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logger
logging.basicConfig(
    filename="logs/resume_matcher.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w"
)

logger = logging.getLogger("resume_matcher")
