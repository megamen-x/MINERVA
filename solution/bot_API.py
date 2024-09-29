import os
import json
import time
import fire
import locale
import logging
import asyncio
import traceback
import numpy as np

# aigram
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.client.telegram import TelegramAPIServer
from aiogram.client.session.aiohttp import AiohttpSession
import requests
from database import Database


class WorkStates(StatesGroup):
    """States for the bot"""
    DEFAULT = State()
    E_MAILING = State()
    TG_MAILING = State()
    SET_EMAIL = State()


class Minerva:
    """
    Bot class Minerva.

    """
    def __init__(
        self,
        bot_token: str,
        db_path: str,
        history_max_tokens: int,
    ):
        """
        Bot initiation
        
        Args:
            bot_token (str): Bot token.
            db_path (str): Path to the database.
            history_max_tokens (int): Maximum number of tokens in history - for the future.
        """
        self.default_prompt = 'Ты бот Минерва, полное имя Богиня Минерва. \nТы отвечаешь от лица женского рода. \nТы бот. \nТы говоришь коротко и емко. \nТы была создана в компании Rutube (она же Рутьюб). \nТы работаешь на компанию Rutube (она же Рутьюб). \nТвое предназначение – отвечать на вопросы, помогать людям. \nТы эксперт в сфере сервисов Rutube.'
        assert self.default_prompt
        self.history_max_tokens = history_max_tokens

        self.db = Database(db_path)

        self.likes_kb = InlineKeyboardBuilder()
        self.likes_kb.add(InlineKeyboardButton(
            text="👍",
            callback_data="feedback:like"
        ))
        self.likes_kb.add(InlineKeyboardButton(
            text="👎",
            callback_data="feedback:dislike"
        ))

        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()

        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.about, Command("about"))
        self.dp.message.register(self.team, Command("team"))
        
        self.dp.message.register(self.generate)
        
        self.dp.callback_query.register(self.save_feedback, F.data.startswith("feedback:"))


    async def start_polling(self):
        """
        Launching the bot.
        """
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message):
        """
        Processing the start command.

        Args:
            message (Message): User message.
        """
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await message.reply("Привет! Меня зовут Minerva, как тебе помочь?")
    
    # Intro: Это MINERVA - интеллектуальный помощник оператора службы поддержки RUTUBE от команды megamen!
    
    async def about(self, message: Message):
        """
        The about command is a short text about the bot.

        Args:
            message (Message): User message.
        """
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await self.bot.send_photo(photo=FSInputFile("Minerva_tg.png"), chat_id=message.chat.id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text="MINERVA - интеллектуальный помощник оператора службы поддержки RUTUBE от команды megamen!",
        )
        
    async def team(self, message: Message):
        """
        Team - a short text about the project team.

        Args:
            message (Message): User message.
        """
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await self.bot.send_photo(photo=FSInputFile("megamen-team.png"), chat_id=message.chat.id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text="""Мы, команда megamen, частые участники хакатонов разного уровня. \n\nНаши проекты это: \n• отличное качество \n• высокие метрики \n• классный дизайн \n\nНадеемся, что данный бот вам будет полезен."""
        )
    
    def get_user_name(self, message: Message):
        """
        Retrieving username.

        Args:
            message (Message): User message.

        Returns:
            str: username.
        """
        return message.from_user.full_name if message.from_user.full_name else message.from_user.username

    async def generate(self, message: Message):
        """
        Generates an answer to the user's question.

        Args:
            message (Message): User message.
        """
        user_id = message.from_user.id
        user_name = self.get_user_name(message)
        chat_id = user_id
        conv_id = self.db.get_current_conv_id(chat_id)

        content = await self._build_content(message)
        if not isinstance(content, str):
            await message.answer("Ошибка! Выбранная модель не может обработать ваше сообщение")
            return
        if content is None:
            await message.answer("Ошибка! Такой тип сообщений пока не поддерживается!")
            return

        self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)
        placeholder = await message.answer("💬")

        try:
            answer = await self.query_api(
                user_content=content,
            )
            markup = self.likes_kb.as_markup()
            new_message = await placeholder.edit_text(answer, parse_mode=None, reply_markup=markup)

            self.db.save_assistant_message(
                content=answer,
                conv_id=conv_id,
                message_id=new_message.message_id,
            )

        except Exception:
            traceback.print_exc()
            await placeholder.edit_text("Что-то пошло не так")


    async def save_feedback(self, callback: CallbackQuery):
        """
        Processing feedback (👍 or 👎).

        Args:
            callback (CallbackQuery): feedback.
        """
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.split(":")[1]
        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )

    @staticmethod
    def _merge_messages(messages):
        """
        Message merge function.

        Args:
            messages (list): List of messages.

        Returns:
            list: Combined list of messages.
        """
        new_messages = []
        prev_role = None
        for m in messages:
            content = m["text"]
            role = m["role"]
            if content is None:
                continue
            if role == prev_role:
                is_current_str = isinstance(content, str)
                is_prev_str = isinstance(new_messages[-1]["text"], str)
                if is_current_str and is_prev_str:
                    new_messages[-1]["text"] += "\n\n" + content
                    continue
            prev_role = role
            new_messages.append(m)
        return new_messages

    def _crop_content(self, content):
        """
        Content trimming function.

        Args:
            content (str): Content.

        Returns:
            str: trimming сontent.
        """
        if isinstance(content, str):
            return content.replace("\n", " ")[:40]
        return "Not text"

    async def query_api(self, user_content):
        """
        Query to the generation model.

        Args:
            user_content (str): User message content.

        Returns:
            str: Model response.
        """
        API_TOKEN = "AQVNyFmd2wp6EBsptTgwLOQ18HDxTpKpNjInMkvH"
        # NODE_ID = "bt1rao4vdhcscdv79b7j"
        NODE_ALIAS = "datasphere.user.megamen"
        FOLDER_ID = "b1g0rlo9stilfpv8erqq"
        BASE_URL = f"https://node-api.datasphere.yandexcloud.net/send/"
        
        data = {
            "question": user_content
        }
        
        headers = {
            # "x-node-id": NODE_ID,
            "x-node-alias": NODE_ALIAS,
            # "Content-Type": "application/json",
            "Authorization": "Api-Key" + API_TOKEN,
            "x-folder-id": FOLDER_ID,
        }
        
        start = time.time()
        try:
            responce = requests.post(BASE_URL, headers=headers, data=json.dumps(data), timeout=1000)
        except:
            responce = ''
        end = time.time()
        print(end - start)
        if responce:
            return json.loads(responce.text)['answer']
        else:
            return 'Что-то не так, ответить не могу! \n(Напишите тех. поддержке @al_goodini)'


    async def _build_content(self, message: Message):
        """
        Content construction.

        Args:
            message (Message):  User message.

        Returns:
            str: Final answer.
        """
        content_type = message.content_type
        if content_type == "text":
            text = message.text
            return text
        
        return None


def main(
    bot_token: str,
    db_path: str,
    history_max_tokens: int = 4500,
) -> None:
    global index, retriever
    bot = Minerva(
        bot_token=bot_token,
        db_path=db_path,
        history_max_tokens=history_max_tokens,
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)