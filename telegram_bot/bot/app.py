from aiogram import Dispatcher

from telegram_bot.bot.router import router

dp = Dispatcher()

dp.include_router(router)
