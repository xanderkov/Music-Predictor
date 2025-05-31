import os
import traceback
import urllib.parse

import aiohttp
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
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
                        genres = "\n".join(map(str, result["genres"]))
                        await message.reply(f"Жанры:\n{genres}")
            except Exception as e:
                await message.reply(f"Жанры не определились: {e}")


@router.message(Command("clear_cache"))
async def clear_cache(message: Message):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{config.backend_url}/api/v1/clear_cache") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Ответ {result}")
                    await message.reply("Кэш очищен.")
        except Exception as e:
            await message.reply(f"Чета не то: {e}")


@router.message(Command("top_genres"))
async def top_genres(message: Message):
    logger.info("Get top genres")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{config.backend_url}/api/v1/top_genres") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Ответ {result}")
                    if not result["top_genres"]:
                        await message.reply("Нет данных о жанрах.")
                        return
                    top_genres_text = ""
                    for genre in result["top_genres"]:
                        top_genres_text += f"*Жанр:* {genre['genre']}\n"
                        top_genres_text += f"*Количество песен:* {genre['count']}\n"
                        top_genres_text += "*Примеры песен:\n"
                        for song in genre["songs"]:
                            song_name = urllib.parse.unquote(song)
                            top_genres_text += f"  - {song_name}\n"
                        top_genres_text += "\n"
                    await message.reply(top_genres_text)
        except Exception as e:
            logger.error(traceback.format_exc())
            await message.reply(f"Чета не то: {e}")


@router.message(F.animation)
async def message_with_gif(message: Message):
    await message.answer("Это кринж!")


@router.message(F.sticker)
async def message_with_sticker(message: Message):
    await message.answer("Это кринж!")


@router.message(Command("help"))
async def help(message: Message):
    logger.info("Help command")
    help_text = "Доступные команды:\n"
    help_text += "/start - Начать работу с ботом\n"
    help_text += "/help - Показать эту справку\n"
    help_text += "/clear_cache - Очистить кэш бота\n"
    help_text += "/top_genres - Показать топ жанров\n"
    help_text += "Загрузка mp3 файла - Определить жанр музыки\n"

    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/start"), KeyboardButton(text="/help")],
            [KeyboardButton(text="/clear_cache"), KeyboardButton(text="/top_genres")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )

    await message.answer(help_text, reply_markup=keyboard)
