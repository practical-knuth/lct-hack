import os
from time import sleep

import telebot
from telebot import types

from tg_bot.src.gigachat_agent import respond
from tg_bot.src.tg_commands_functions import *

bot_token = os.environ["BOT_TOKEN"]

bot = telebot.TeleBot(bot_token)

# Список разрешенных пользователей (может быть user_id или username)
allowed_users = []


# Функция для проверки, авторизован ли пользователь
def is_authorized(user_id, username):
    return True


# Функция для автоматического ответа в случае нетекстового сообщения
@bot.message_handler(
    content_types=[
        "audio",
        "video",
        "document",
        "photo",
        "sticker",
        "voice",
        "location",
        "contact",
    ]
)
def not_text(message):
    user_id = message.chat.id
    bot.send_message(user_id, "Я работаю только с текстовыми сообщениями!")


# Обработчик команды /start
@bot.message_handler(commands=["start"])
def send_welcome(message):
    user_id = message.from_user.id
    username = message.from_user.username

    if is_authorized(user_id, username):
        bot.reply_to(message, "Добро пожаловать! Вы авторизованы.")
    else:
        bot.reply_to(message, "Извините, у вас нет доступа к этому боту.")


import datetime


# Обработчик команды /help
@bot.message_handler(commands=["help"])
def handle_start(message):
    help_text = """ Привет, я меня зовут Асистент Закупок. 
    Я могу выводить названия товаров с которыми работаю, делать прогноз по ним и показывать остатки на последнею дату.
    Еще я могу выгрузить все прогнозы в файл и отправить его в сообщения.
    Я понимаю естественную речь, но иногда могу делать ошибки.
    Я только учусь, поэтому иногда могу не понимать что нужно сделать, если я не отвечаю неправильно,
    попробуйте перефразировать вопрос или воспользуйтесь шаблонными командами.
    Со временем я буду развиваться и лучше понимать человеческую речь!
    """
    bot.reply_to(message, help_text)


# Обработчик команды /get_product_name
@bot.message_handler(commands=["get_product_name"])
def handle_start(message):
    msg = bot.reply_to(message, "Введите количество товаров для вывода:")
    bot.register_next_step_handler(msg, process_product_name_step)


def process_product_name_step(message):
    try:
        parameter = int(message.text)  # Пробуем преобразовать ввод в число
        product_names = show_product_names(parameter)
        bot.reply_to(message, product_names)
    except:
        bot.reply_to(
            message, "Ошибка! Введите число. Вызовите команду еще раз /get_product_name"
        )


# Обработчик команды /get_product_stocks
@bot.message_handler(commands=["get_product_stocks"])
def handle_start(message):
    msg = bot.reply_to(message, "Введите точное название товара:")
    bot.register_next_step_handler(msg, process_product_stocks_step)


def process_product_stocks_step(message):
    try:
        product_names = stocks_info(message.text)
        bot.reply_to(message, product_names)
    except:
        bot.reply_to(
            message,
            """Ошибка! У меня нет такого товара, проверьте корректность ввода или попробуйте другой товар. 
                                 Вызовите команду еще раз /get_product_stocks""",
        )


# Обработчик команды /get_product_forecast
forecast_params = {}


@bot.message_handler(commands=["get_product_forecast"])
def handle_get_product_name(message):
    chat_id = message.chat.id
    if not os.path.exists(f"tg_bot/chats/{chat_id}"):
        os.mkdir(f"tg_bot/chats/{chat_id}")
    msg = bot.reply_to(message, "Выберите горизонт прогнозирования (1-12 месяцев):")
    bot.register_next_step_handler(msg, process_first_param_step)


def process_first_param_step(message):
    try:
        chat_id = message.chat.id
        assert int(message.text) in range(1, 13)
        forecast_params[chat_id] = {"horizon": message.text}

        msg = bot.reply_to(message, "Введите название продукта:")
        bot.register_next_step_handler(msg, process_second_param_step)
    except:
        bot.reply_to(
            message,
            """Горизонт прогнозирвоания должен быть числом в диапазоне 1-12.
                                 Вызовите команду еще /get_product_forecast еще раз""",
        )


def process_second_param_step(message):
    try:
        chat_id = message.chat.id
        forecast_params[chat_id]["prod_name"] = message.text
        params = forecast_params[chat_id]
        result = get_forecast(params["prod_name"], chat_id, params["horizon"])
        with open(f"tg_bot/chats/{chat_id}/forecast_chart.png", "rb") as photo:
            bot.send_photo(chat_id, photo)
        bot.send_message(chat_id, result)
    except:
        bot.reply_to(message, '''Ошибка! У меня нет такого товара, проверьте корректность ввода или попробуйте другой товар.
                                 Вызовите команду еще /get_product_forecast еще раз''')


# Обработчик команды /download_all_forecasts
@bot.message_handler(commands=["download_all_forecasts"])
def handle_start(message):
    chat_id = message.chat.id
    download_forecast(chat_id)
    bot.send_document(chat_id, open(f"tg_bot/chats/{chat_id}/prediction.json", "rb"))


# Обработчик команды /get_recommendation
@bot.message_handler(commands=["get_recommendation"])
def handle_start(message):
    chat_id = message.chat.id
    rec = download_recommendation(chat_id)
    if len(rec[rec < 0]) != 0:
        bot.send_message(chat_id, f"Нужно закупить следующие товары: {rec[rec<0]}")
        bot.send_document(
            chat_id, open(f"tg_bot/chats/{chat_id}/recommendation.json", "rb")
        )
    else:
        bot.send_message(chat_id, "Нет потребностей в закупках в ближайший год.")


# Функция, обрабатывающая текстовые сообщения
@bot.message_handler(content_types=["text"])
def handle_text_message(message):
    username = message.from_user.username
    user_id = message.chat.id
    if not os.path.exists(f"tg_bot/chats/{user_id}"):
        os.mkdir(f"tg_bot/chats/{user_id}")
    if is_authorized(user_id, username):
        try:
            res = respond(message=message.text, user_id=user_id)
            gigachat_response = res["output"]
            bot.send_message(user_id, gigachat_response)
            sleep(2)

            c1 = types.BotCommand(command="help", description="Нажми для помощи.")
            c2 = types.BotCommand(
                command="get_product_name", description="Увидеть названиия товаров."
            )
            c3 = types.BotCommand(
                command="get_product_stocks",
                description="Посмотреть остатки на последнюю дату по определенному товару.",
            )
            c4 = types.BotCommand(
                command="get_product_forecast",
                description="Сделать прогноз по определенному товару.",
            )
            c5 = types.BotCommand(
                command="download_all_forecasts",
                description="Скачать прогнозы по всем товарам в формате json.",
            )
            c6 = types.BotCommand(
                command="get_recommendation",
                description="Получить рекомендации по закупкам.",
            )

            bot.set_my_commands([c1, c2, c3, c4, c5, c6])
            bot.set_chat_menu_button(
                message.chat.id, types.MenuButtonCommands("commands")
            )
        except:
            bot.send_message(
                user_id, "Извините, произошла ошибка, повторите, пожалуйста, запрос"
            )
            sleep(2)
    else:
        bot.reply_to(message, "Извините, у вас нет доступа к этому боту.")


def run_bot():
    bot.polling(none_stop=True)
