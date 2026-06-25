import os
from datetime import datetime
import logging


# 1. Directory and File Path Setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_BASE_NAME = f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_BASE_NAME)

# 2. Advanced Formatter (Includes milliseconds for timing and clear module structure)
# Use exact format to track LLM latency and pipeline steps
LOG_FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 3. Configure the Root Logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # Writes to your timestamped log file
        logging.StreamHandler()             # Prints to your terminal
    ]
)
