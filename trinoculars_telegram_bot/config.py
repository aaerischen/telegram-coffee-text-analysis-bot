import os
import sys
from pathlib import Path


# Project root: .../tg bot
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Path to the Trinoculars-main project
TRINOCULARS_PATH = PROJECT_ROOT / "Trinoculars-main"

# Ensure Trinoculars is importable (model_utils, text_analysis, etc.)
if str(TRINOCULARS_PATH) not in sys.path:
    sys.path.insert(0, str(TRINOCULARS_PATH))

# Telegram bot token (prefer setting via environment variable)
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "YOUR_TG_BOT_TOKEN",
)

# Minimal length of text to run classification
MIN_TEXT_LENGTH = 100

# Default model settings for Trinoculars
# model_type: 'binary' or 'three-class'
MODEL_TYPE = os.getenv("TRINOCULARS_MODEL_TYPE", "binary")

# Optional explicit directory with weights; if empty, Trinoculars defaults are used
# For binary in this repo, typically: 'models/medium_binary_classifier'
MODEL_DIR = os.getenv("TRINOCULARS_MODEL_DIR", "models/medium_binary_classifier")

# Whether to use Binoculars scores (score_chat, score_coder) as in CLI demo
# WARNING: this loads large models and will be slow on CPU.
USE_BINOCULARS = os.getenv("TRINOCULARS_USE_BINOCULARS", "false").lower() in {
    "1",
    "true",
    "yes",
}
