import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import telebot
from langchain_core.tools import tool

from modeling.src.preprocess_data import create_date_suffix
from tg_bot.constants import constants

matplotlib.use("agg")


bot_token = os.environ["BOT_TOKEN"]
bot = telebot.TeleBot(bot_token)


@tool
def stocks_info(product_name: str) -> int:
    """
    Выводит складские остатки по товару

    Args:
        product_name (str): название товара

    Returns:
        int: количество товара на складе
    """
    suffix = create_date_suffix()

    df = pd.read_parquet(
        os.path.join(constants.stock_path, f"stocks__aggregated_{suffix}.parquet")
    )

    if product_name.lower() not in df[constants.product_level_name].unique():
        return (
            "Скажи пользовтелю что данного товара нет в базе."
            "Вызови функцию show_product_names и выведи названия товаров."
            "Предупреди пользователя, что тебе нужно точное название товара."
            "Предложи заключить назание товара в кавычки."
        )

    return df[df[constants.product_level_name] == product_name.lower()][
        constants.stock_name
    ].values[0]


@tool
def show_product_names(chat_id: str, N: int = 5) -> list:
    """
    Функция выводит на экран пользователю список товаров.
    Args:
        chat_id (str): индентификатор пользователя с которым ты общаешься, подставь его из системных настроек, не спрашивай его у пользователя.
        N (int): количестов товаров для вывода.
    Returns:
        str: Уведомление для пользователя, что список выведен.

    """
    if not chat_id:
        return "Возьми идентификатор пользователя с которым ты общаешься, подставь его из системных настроек. Потом вызови эту функцию снова"

    suffix = create_date_suffix()

    df = pd.read_parquet(
        os.path.join(constants.stock_path, f"stocks__aggregated_{suffix}.parquet")
    )

    product_list = "\n -".join(np.random.choice(
        df[constants.product_level_name].unique(), N, replace=False
    ))

   
    text = (
        f"Вот список {N} случайных товаров, которые есть в базе данных:\n -{product_list}"
    ) 

    bot.send_message(chat_id, text)

    return 'Напиши только это: "Список товаров выведен."'



@tool
def get_forecast(product_name: str, chat_id: str, N: int = 12):
    """
    Делает прогноз и рекомендацию по закупкам на основе прогноза и остатков.

    Args:
        product_name (str): название товара
        chat_id (str): индентификатор пользователя с которым ты общаешься, подставь его из систменых настроек, не спрашивай его у пользователя.
        N (str): на сколько месяцев делается прогноз

    Returns:
        прогноз, график прогноза и рекомендации по закупкам
    """
    
    if not chat_id:
        return "Возьми идентификатор пользователя с которым ты общаешься, подставь его из систменых настроек. Потом вызови эту функцию снова"

    suffix = create_date_suffix()

    target_df = pd.read_parquet(
        os.path.join(constants.target_path, f"target__aggregated_{suffix}.parquet")
    )
    stocks_df = pd.read_parquet(
        os.path.join(constants.stock_path, f"stocks__aggregated_{suffix}.parquet")
    )
    prediction_df = pd.read_parquet(
        os.path.join(constants.prediction_path, f"predictions_{suffix}.parquet")
    )

    product_name = product_name.lower()

    

    if product_name not in target_df[constants.product_level_name].unique():
        return (
            "Скажи пользователю что данного товара нет в базе."
            "Вызови функцию show_product_names и выведи названия товаров."
            "Предупреди пользователя, что тебе нужно точное название товара."
            "Предложи заключить назание товара в кавычки."
            
        )

    target_df = target_df[(target_df[constants.product_level_name] == product_name)]

    stocks_df = stocks_df[(stocks_df[constants.product_level_name] == product_name)]

    prediction_df = prediction_df[
        prediction_df[constants.product_level_name] == product_name
    ]

    fig, ax1 = plt.subplots(figsize=(16, 6))
    plt.grid()
    ax2 = ax1.twinx()

    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Закупки, шт")
    ax2.set_ylabel("Остатки, шт")

    ax2.bar(
        pd.to_datetime(stocks_df[constants.date_name]),
        stocks_df[constants.stock_name],
        label="Остатки",
        width=3,
    )

    ax1.plot(
        pd.to_datetime(target_df[constants.date_name]),
        target_df[constants.target_name],
        label="Факт",
        color="green",
    )

    ax1.plot(
        pd.to_datetime(prediction_df[constants.date_name]),
        prediction_df[constants.prediction_name].values[:int(N)],
        label="Прогноз",
        color="orange",
    )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.title(f"Товар - {product_name}")
    plt.savefig(f"tg_bot/chats/{chat_id}/forecast_chart.png")

    with open(f"tg_bot/chats/{chat_id}/forecast_chart.png", "rb") as photo:
        bot.send_photo(chat_id, photo)

    stocks_analyze = round(
        (
            stocks_df[constants.stock_name].values[-1]
            - prediction_df[constants.prediction_name].sum()
        )
    )

    if stocks_analyze < 0:
        bot.send_message(
            chat_id,
            f"""
        Вам нужно закупить {abs(stocks_analyze)} штук и 10% - {round(abs(stocks_analyze)*0.1)} страховой запас.
        Сейчас осталось {stocks_df[constants.stock_name].values[-1]} штук товара {product_name}.
        В ближайшие {N} месяцев нужно купить {round(prediction_df[constants.prediction_name].sum())} штук товара.
        """,
        )

    else:
        bot.send_message(
            chat_id,
            f"""
        У вас достаточно товара.
        Сейчас осталось {stocks_df[constants.stock_name].values[-1]} штук товара {product_name}.
        В ближайшие {N} месяцев нужно купить {round(prediction_df[constants.prediction_name].sum())} штук товара.
        """,
        )

    return 'Напиши только это: "Прогноз выполнен."'


@tool
def download_forecast(chat_id: int) -> list:
    """
    Функция выгружает пользователю файл с прогнозами по всем товарам.
    Args:
        chat_id (str): индентификатор пользователя с которым ты общаешься, подставь его из систменых настроек, не спрашивай его у пользователя.
    Returns:
        str: информация что файл отправлен
    """
    suffix = create_date_suffix()

    target_df = pd.read_parquet(
        os.path.join(constants.target_path, f"target__aggregated_{suffix}.parquet")
    )
    stocks_df = pd.read_parquet(
        os.path.join(constants.stock_path, f"stocks__aggregated_{suffix}.parquet")
    )
    prediction_df = pd.read_parquet(
        os.path.join(constants.prediction_path, f"predictions_{suffix}.parquet")
    )

    if not chat_id:
        return "Возьми идентификатор пользователя с которым ты общаешься, подставь его из систменых настроек. Потом вызови эту функцию снова"

    result = {}

    # Группировка и объединение данных по product
    for product in target_df[constants.product_level_name].unique():
        fact_group = target_df[target_df[constants.product_level_name] == product]
        remains_group = stocks_df[stocks_df[constants.product_level_name] == product]
        prediction_group = prediction_df[
            prediction_df[constants.product_level_name] == product
        ]

        result[product] = {
            "fact": [
                fact_group[constants.target_name].tolist(),
                fact_group[constants.date_name].astype(str).tolist(),
            ],
            "remains": [
                remains_group[constants.stock_name].tolist(),
                remains_group[constants.date_name].astype(str).tolist(),
            ],
            "predictions": [
                prediction_group[constants.prediction_name].tolist(),
                prediction_group[constants.date_name].astype(str).tolist(),
            ],
        }

    with open(f"tg_bot/chats/{chat_id}/prediction.json", "w") as outfile:
        json.dump(result, outfile)

    bot.send_document(chat_id, open(f"tg_bot/chats/{chat_id}/prediction.json", "rb"))

    return "файл отправлен"


@tool
def download_recommendation(chat_id: int) -> str:
    """
    Функция выводит рекомендации по закупкам для товаров.
    Args:
        chat_id (str): индентификатор пользователя с которым ты общаешься, подставь его из систменых настроек, не спрашивай его у пользователя.
    Returns:
        str: сообщает пользователю нужно делать закупку или нет
    """

    if not chat_id:
        return "Возьми идентификатор пользователя с которым ты общаешься, подставь его из систменых настроек. Потом вызови эту функцию снова"

    suffix = create_date_suffix()

    stocks_df = pd.read_parquet(
        os.path.join(constants.stock_path, f"stocks__aggregated_{suffix}.parquet")
    )
    prediction_df = pd.read_parquet(
        os.path.join(constants.prediction_path, f"predictions_{suffix}.parquet")
    )

    result = (
        stocks_df.groupby(constants.product_level_name)[constants.stock_name].last()
        - prediction_df.groupby(constants.product_level_name)[
            constants.prediction_name
        ].sum()
    )

    with open(f"tg_bot/chats/{chat_id}/recommendation.json", "w") as outfile:
        json.dump(result.to_json(), outfile)

    if len(result[result < 0]) != 0:
        return f"Скажи пользователю, что нужно закупить следующие товары: {result[result<0]}"

    else:
        return "Товаров хватит на 12 месяцев, ничего покупать не нужно"


new_tools = [
    stocks_info,
    show_product_names,
    get_forecast,
    download_forecast,
    download_recommendation,
]
