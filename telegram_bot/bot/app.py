from aiogram import Bot, Dispatcher

from telegram_bot.bot.config_reader import config

bot = Bot(token=config.bot_token.get_secret_value())
dp = Dispatcher()

dp.include_router(dp)
