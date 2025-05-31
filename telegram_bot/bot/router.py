import os

import aiohttp
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import Message
from loguru import logger

from telegram_bot.bot.config_reader import config

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info("Start command")
    await message.answer(
        "Приветствую, я бот предоставляющий жанры загруженной в меня музыки."
    )


@router.message(F.audio)
async def handle_audio(message: Message, bot: Bot):
    logger.info("handle audio")
    file = await bot.get_file(message.audio.file_id)
    file_path = file.file_path
    destination = f"downloads/{message.audio.file_name}"
    file_name = message.audio.file_name or "audio.mp3"
    os.makedirs("downloads", exist_ok=True)
    await bot.download_file(file_path, destination)
    logger.info("File downloaded")
    # audio = FSInputFile(destination)
    # await bot.send_audio(
    #     chat_id=message.chat.id,
    #     audio=audio,
    #     caption="Это ваш аудиофайл!",
    #     performer=message.audio.performer,
    #     title=message.audio.title,
    # )
    async with aiohttp.ClientSession() as session:
        with open(destination, "rb") as f:
            form = aiohttp.FormData()
            form.add_field(
                name="music_file",
                value=f,
                filename=file_name,
                content_type="audio/mpeg",
            )
            try:
                async with session.post(
                    f"{config.backend_url}/api/v1/predict_mp3", data=form
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"Жанры: {result}")
                        await message.reply(f"Жанры: {result['genres']}")
            except Exception as e:
                await message.reply(f"Жанры не определились: {e}")


@router.message(F.animation)
async def message_with_gif(message: Message):
    await message.answer("Это кринж!")


@router.message(F.sticker)
async def message_with_sticker(message: Message):
    await message.answer("Это кринж!")
