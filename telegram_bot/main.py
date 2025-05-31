import asyncio

from aiogram import Bot

from telegram_bot.bot.app import dp
from telegram_bot.bot.config_reader import config


async def main():
    bot = Bot(token=config.bot_token.get_secret_value())

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
