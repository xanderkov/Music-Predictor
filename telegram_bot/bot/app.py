from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command

from telegram_bot.bot.config_reader import config

bot = Bot(token=config.bot_token.get_secret_value())
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Приветствую, я бот предоставляющий жанры загруженной в меня музыки."
    )
