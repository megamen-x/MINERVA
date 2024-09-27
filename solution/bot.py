import os
import json
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
from database import Database

# document processing
"""
from io import BytesIO
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple
from dotenv import load_dotenv
from infer import get_information
from emaling import send_bulk_email
from punctuation_spell import update_punctuation
from docx_pdf import convert_words_to_pdfs, add_encryption
from pattern import fill_decrypton, fill_official, fill_unofficial
"""


# States for the bot
class WorkStates(StatesGroup):
    DEFAULT = State()
    E_MAILING = State()
    TG_MAILING = State()
    SET_EMAIL = State()


class Minerva:
    def __init__(
        self,
        bot_token: str,
        db_path: str,
        history_max_tokens: int,
        chunk_size: int,
    ):
        self.default_prompt = 'Ты бот Минерва, полное имя Богиня Минерва. \nТы отвечаешь от лица женского рода. \nТы бот. \nТы говоришь коротко и емко. \nТы была создана в компании Rutube (она же Рутьюб). \nТы работаешь на компанию Rutube (она же Рутьюб). \nТвое предназначение – отвечать на вопросы, помогать людям. \nТы эксперт в сфере сервисов Rutube.'
        assert self.default_prompt

        # Параметры
        self.history_max_tokens = history_max_tokens
        self.chunk_size = chunk_size

        # База
        self.db = Database(db_path)

        # Клавиатуры
        self.likes_kb = InlineKeyboardBuilder()
        self.likes_kb.add(InlineKeyboardButton(
            text="👍",
            callback_data="feedback:like"
        ))
        self.likes_kb.add(InlineKeyboardButton(
            text="👎",
            callback_data="feedback:dislike"
        ))

        # Бот
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()

        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.about, Command("about"))
        self.dp.message.register(self.team, Command("team"))
        self.dp.message.register(self.reset, Command("reset_history"))
        self.dp.message.register(self.history, Command("history"))
        self.dp.message.register(self.generate)

        self.dp.callback_query.register(self.save_feedback, F.data.startswith("feedback:"))


    async def start_polling(self):
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message):
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await message.reply("Привет! Меня зовут Minerva, как тебе помочь?")
    
    # Intro: Это MINERVA - интеллектуальный помощник оператора службы поддержки RUTUBE от команды megamen!
    
    async def about(self, message: Message):
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await self.bot.send_photo(photo=FSInputFile("Minerva_tg.png"), chat_id=message.chat.id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text="MINERVA - интеллектуальный помощник оператора службы поддержки RUTUBE от команды megamen!",
        )
        
    async def team(self, message: Message):
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await self.bot.send_photo(photo=FSInputFile("megamen-team.png"), chat_id=message.chat.id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text="""Мы, команда megamen, частые участники хакатонов разного уровня. \n\nНаши проекты это: \n• отличное качество \n• высокие метрики \n• классный дизайн \n\nНадеемся, что данный бот вам будет полезен."""
        )

    async def reset(self, message: Message):
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await message.reply(f"История сообщений сброшена!, {chat_id}")

    async def history(self, message: Message):
        chat_id = message.chat.id
        conv_id = self.db.get_current_conv_id(chat_id)
        history = self.db.fetch_conversation(conv_id)
        for m in history:
            if not isinstance(m["text"], str):
                m["text"] = "Not text"
        history = self._merge_messages(history)
        history = json.dumps(history, ensure_ascii=False)
        if len(history) > self.chunk_size:
            history = history[:self.chunk_size] + "... truncated"
        await message.reply(history, parse_mode=None)

    def get_user_name(self, message: Message):
        return message.from_user.full_name if message.from_user.full_name else message.from_user.username

    async def generate(self, message: Message):
        user_id = message.from_user.id
        user_name = self.get_user_name(message)
        chat_id = user_id
        conv_id = self.db.get_current_conv_id(chat_id)
        history = self.db.fetch_conversation(conv_id)
        system_prompt = self.db.get_system_prompt(chat_id, self.default_prompt)

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
            answer, _ = await self.query_api(
                history=history,
                user_content=content,
                system_prompt=system_prompt
            )
            chunk_size = self.chunk_size
            answer_parts = [answer[i:i + chunk_size] for i in range(0, len(answer), chunk_size)]
            markup = self.likes_kb.as_markup()
            new_message = await placeholder.edit_text(answer_parts[0], parse_mode=None, reply_markup=markup)

            self.db.save_assistant_message(
                content=answer,
                conv_id=conv_id,
                message_id=new_message.message_id,
                system_prompt=system_prompt
            )

        except Exception:
            traceback.print_exc()
            await placeholder.edit_text("Что-то пошло не так")


    async def save_feedback(self, callback: CallbackQuery):
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
        if isinstance(content, str):
            return content.replace("\n", " ")[:40]
        return "Not text"

    async def query_api(self, history, user_content, system_prompt: str) -> tuple:
        # messages = history + [{"role": "user", "text": user_content}]
        # messages = self._merge_messages(messages)
        # assert messages

        # if messages[0]["role"] != "system" and system_prompt.strip():
        #     messages.insert(0, {"role": "system", "text": system_prompt})
        # assert messages
        
        # urls, chunks, relevant_score = top_k_rerank(user_content, retriever, reranker)


        # messages[1]['text'] = '''Используй только следующий контекст, чтобы кратко ответить на вопрос в конце.
        #     Не пытайся выдумывать ответ. Если контекст не соотносится с вопросом, скажи, что ты не можешь ответить на данный вопрос
        #     Контекст:
        #     ===========
        #     {texts}
        #     ===========
        #     Вопрос:
        #     ===========
        #     {query}'''.format(texts='', query=user_content)
        # chunks
        # if relevant_score >= 0.805:
        #     answer = response(messages)
        #     formatted_answer = '{llm_gen}\n===================================\nСсылки с дополнительной информацией:\n1. {url_1}\n2. {url_2}\n3. {url_3}\n'.format(llm_gen=answer['result']['alternatives'][0]['message']['text'],
        #     url_1=urls[0], url_2=urls[1], url_3=urls[2])
        #     return formatted_answer, urls
        # else:
        #     return 'Данный вопрос выходит за рамки компетенций бота. Пожалуйста, переформулируйте вопрос или попросите вызвать сотрудника.'
        return 'Что-то не так, ответить не могу! (Напишите тех. поддержке @al_goodini)', ['kek.ru', 'lol.com']


    async def _build_content(self, message: Message):
        content_type = message.content_type
        if content_type == "text":
            text = message.text
            return text
        
        return None


def main(
    bot_token: str,
    db_path: str,
    history_max_tokens: int = 4500,
    chunk_size: int = 2000,
) -> None:
    global index, retriever
    # index, retriever = start_rag()
    bot = Minerva(
        bot_token=bot_token,
        db_path=db_path,
        history_max_tokens=history_max_tokens,
        chunk_size=chunk_size,
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)