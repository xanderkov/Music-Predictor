from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from loguru import logger

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info("Start command")
    await message.answer(
        "Приветствую, я бот предоставляющий жанры загруженной в меня музыки."
    )
