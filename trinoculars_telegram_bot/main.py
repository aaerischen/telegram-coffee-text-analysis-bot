"""
Run this file to start the bot:

    cd path\to\folder
    python -m trinoculars_telegram_bot.main
or
    python trinoculars_telegram_bot/main.py
"""

import asyncio
import logging

from .bot import run_bot


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    setup_logging()
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Bot stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logging.getLogger(__name__).exception("Fatal error in bot: %s", e)


if __name__ == "__main__":
    main()


