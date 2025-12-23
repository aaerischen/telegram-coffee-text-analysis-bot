import logging
from typing import Optional

from aiogram import Bot, Dispatcher

from .config import TELEGRAM_BOT_TOKEN
from .handlers import router as handlers_router
from .model_service import load_detection_model

logger = logging.getLogger(__name__)


async def run_bot(token: Optional[str] = None) -> None:
    actual_token = token or TELEGRAM_BOT_TOKEN
    if not actual_token or actual_token == "PASTE_YOUR_TOKEN_HERE":
        logger.error(
            "Telegram token is not configured. "
            "Set TELEGRAM_BOT_TOKEN environment variable or edit config.py."
        )
        return

    if not load_detection_model():
        logger.error("Failed to load Trinoculars model. Bot will not start.")
        return

    bot = Bot(token=actual_token)
    dp = Dispatcher()

    # Attach all handlers
    dp.include_router(handlers_router)

    logger.info("Starting Trinoculars Telegram bot polling...")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        logger.info("Bot session closed.")


