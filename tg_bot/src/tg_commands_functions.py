import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling.src.preprocess_data import create_date_suffix
from tg_bot.constants import constants


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

    product_name = str(product_name)

    stocks = df[df[constants.product_level_name] == product_name.lower()][
        constants.stock_name
    ].values[0]
    return f"Товара '{product_name}' осталось {stocks} штук."


def show_product_names(N: int = 5) -> list:
    """
    Функция выводит на экран пользователю список товаров.

    Args:
        N (int): количестов товаров для вывода
    Returns:
        list: Список с названиями товаров.

    """

    suffix = create_date_suffix()

    df = pd.read_parquet(
        os.path.join(constants.stock_path, f"stocks__aggregated_{suffix}.parquet")
    )

    product_list = "\n -".join(
        np.random.choice(df[constants.product_level_name].unique(), N, replace=False)
    )
    text = (
        f"Вот список {N} случайных товаров, которые есть в базе данных:\n{product_list}"
    )
    return text


def get_forecast(product_name: str, chat_id: str, N: int = 12):
    """
    Делает прогноз и рекомендацию по закупкам на основе прогноза и остатков.

    Args:
        product_name (str): название товара
        chat_id (str): индентификатор пользователя с которым ты общаешься, подставь его из систменых настроек, не спрашивай его у пользователя.
        N (str): горизонт прогнозирования

    Returns:
        прогноз, график прогноза и рекомендации по закупкам
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

    product_name = product_name.lower()

    target_df = target_df[
        (target_df[constants.product_level_name] == product_name)
    ].reset_index(drop=True)

    stocks_df = stocks_df[
        (stocks_df[constants.product_level_name] == product_name)
    ].reset_index(drop=True)

    prediction_df = prediction_df[
        prediction_df[constants.product_level_name] == product_name
    ].reset_index(drop=True)
    prediction_df = prediction_df.loc[0:N]

    fig, ax1 = plt.subplots(figsize=(16, 6))
    plt.grid()
    ax2 = ax1.twinx()

    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Продажи, шт")
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
        prediction_df[constants.prediction_name],
        label="Прогноз",
        color="orange",
    )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")

    plt.title(f"Товар - {product_name}")
    plt.savefig(f"tg_bot/chats/{chat_id}/forecast_chart.png")
    plt.close()

    stocks_analyze = round(
        (
            stocks_df[constants.stock_name].values[-1]
            - prediction_df[constants.prediction_name].sum()
        )
    )

    if stocks_analyze < 0:
        return f"""
        Вам нужно закупить {abs(stocks_analyze)} штук и 10% - {round(abs(stocks_analyze)*0.1)} страховой запас.
        Сейчас осталось {stocks_df[constants.stock_name].values[-1]} штук товара {product_name}.
        В ближайшие {N} месяцев нужно купить {round(prediction_df[constants.prediction_name].sum())} штук товара.
        """

    else:
        return f"""
        У вас достаточно товара.
        Сейчас осталось {stocks_df[constants.stock_name].values[-1]} штук товара {product_name}.
        В ближайшие {N} месяцев нужно купить {round(prediction_df[constants.prediction_name].sum())} штук товара.
        """


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
        os.path.join(constants.target_path), f"target__aggregated_{suffix}.parquet"
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


def download_recommendation(chat_id: int) -> int:
    """
    Выводит складские остатки по товару

    Args:
        product_name (str): название товара

    Returns:
        int: количество товара на складе
    """

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

    return result
