"""

1. Парсинг сырых данных по остаткам
2. Отбор сч 101 и 21, так как в 105 нет даты приема
3. Интерпретируем дату вводу в эксплуатацию как дату спроса на продукт
4. Интерпретируем количество введеного продукта в эксплуатацию как спрос на продукт
5. Отбираем только продукты по которым есть данные по "Длительность полезного использования"
    так как по нему можно рассчитать когда продукт выведут из использования, соответсвенно
        корретно посчитать остатки продукта на кажду дату:
            - кумсум введенных продуктов в эксплуатацию - число выводимых в дату продуктов
6. Понижаем типы данных и сохраняем сагрегированные датафреймы.

"""

import os

import pandas as pd

from modeling.conf import modeling_paths, raw_paths
from modeling.src.preprocess_data import (
    check_create_path,
    create_date_suffix,
    decrease_dtypes,
    fill_skips_in_date,
)


def prepare_refs():
    """
    Предобработка справочника.
    """
    cats = pd.read_excel(raw_paths.references)

    cats_keep_cols = ["product_name", "kpgz"]

    for col, new_name in zip(["Название СТЕ", "КПГЗ"], cats_keep_cols):
        cats = cats.rename(columns={col: new_name})

        try:
            cats[new_name] = cats[new_name].str.strip().str.lower()
        except AttributeError:
            continue

    cats = cats[cats_keep_cols]
    cats = cats.dropna()

    return cats


def parse_stocks():
    """
    Приведение данных с остатками к единому формату.
    """
    drop_cols = [
        "Unnamed: 0",
        "Unnamed: 1",
        "Unnamed: 3",
        "Unnamed: 4",
        "Unnamed: 5",
        "Unnamed: 6",
        "Unnamed: 7",
        "Unnamed: 8",
        "Unnamed: 11",
    ]

    cols_105 = ["Номенклатура", "Цена", "Количество", "Сумма"]

    data_101 = []
    data_105 = []

    for idx, file in enumerate(os.listdir(raw_paths.stocks)):
        stock = pd.read_excel(os.path.join(raw_paths.stocks, file))

        if "105" not in file:
            # дергаем названия колонок
            second = stock.iloc[2, 19:].tolist()
            first = [stock.iloc[7, 2]] + stock.iloc[7, 9:-4].dropna().tolist()
            columns = first + second

            # дергаем тело файла
            stock = stock[~stock.iloc[:, 2].isna()].reset_index(drop=True)
            stock = stock.iloc[1:, :]

            # дропаем неинформативные колонки
            stock = stock.drop(drop_cols, axis=1)

            # переименовываем колонки
            stock.columns = columns

            stock["Основное средство"] = (
                stock["Основное средство"].str.strip().str.lower()
            )

            data_101.append(stock)

        else:
            stock = stock.iloc[8:-1, :].reset_index(drop=True)

            stock = stock.dropna()
            stock.columns = cols_105

            stock["Номенклатура"] = stock["Номенклатура"].str.strip().str.lower()

            data_105.append(stock)

    data = pd.concat(data_101)

    return data


def fill_nans_mode(data: pd.DataFrame, groupby: str, nan_col: str) -> pd.DataFrame:
    """
    Заполнение пропусков модой в категории

    :param data: pd.DataFrame для трансформации
    :param groupby: str катег. колонка внутри которых производится поиск моды
    :param nan_col: str колонка для заполнения пропусков

    :return: pd.DataFrame
    """
    fillnaa = (
        data.groupby(groupby)
        .apply(lambda x: x[nan_col].mode())
        .reset_index()
        .rename(columns={0: nan_col})
    )

    data = data.drop(nan_col, axis=1).merge(fillnaa)

    return data


def preprocess_stocs(data: pd.DataFrame):
    """
    Предобработка остатков

    :param data: pd.DataFrame для трансформации

    :return: pd.DataFrame
    """
    keep_cols = [
        "product_name",
        "date_start",
        "usage_duration",
        "price",
        "counts",
        "status",
        "depreciation_group",
    ]

    for col, new_name in zip(
        [
            "Основное средство",
            "Дата принятия к учету",
            "Срок полезного использования",
            "Балансовая стоимость",
            "Количество",
            "Состояние",
            "Амортизационная группа",
        ],
        keep_cols,
    ):
        data = data.rename(columns={col: new_name})

        try:
            data[new_name] = data[new_name].str.strip().str.lower()
        except AttributeError:
            continue

    data = data[keep_cols]
    data["date_start"] = pd.to_datetime(data["date_start"], dayfirst=True)
    data["price_sum"] = data["price"] * data["counts"]
    data = data.sort_values(by=["product_name", "date_start"])

    # fill nans in usage duration
    data = fill_nans_mode(
        data=data, nan_col="usage_duration", groupby="depreciation_group"
    )

    # add end of product usage date
    data["date_end"] = data["date_start"] + pd.to_timedelta(
        data["usage_duration"] * 30, unit="d"
    )

    return data


def create_target_data(
    data: pd.DataFrame,
    predicting_unit: str,
    date_col: str,
    target: str,
    freq: str,
    groupby_cols: str,
) -> pd.DataFrame:
    """
    Создание датафрейма с прогнозируемой целевой переменной

    :param data: pd.DataFrame для трансформации
    :param predicting_unit: str катег. колонка c кодом прогнозируемых единиц
    :param date_col: str колонка с датой
    :param target: str колонка с целевой переменной
    :param freq: str гранулярность данных
    :param groupby_cols: list колонки для группировки данных

    :return: pd.DataFrame
    """
    data = fill_skips_in_date(
        data=data,
        date_col=date_col,
        target=target,
        groupby_cols=groupby_cols,
        freq=freq,
    )

    data = (
        data.groupby([predicting_unit, pd.Grouper(key=date_col, freq="M")])[target]
        .sum()
        .reset_index()
        .rename(columns={date_col: "date"})
    )

    return data


def create_stocks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Создание датафрейма с остатками

    :param data: pd.DataFrame для трансформации

    :return: pd.DataFrame
    """
    stocks = (
        data.groupby(["product_name", "date_start", "date_end"])["counts"]
        .sum()
        .reset_index()
        .sort_values(by=["product_name", "date_start"])
    )

    # create stocks incoming dataframe
    stocks_income = stocks[["product_name", "date_start", "counts"]].rename(
        columns={"date_start": "date", "counts": "stock_in"}
    )
    stocks_income = fill_skips_in_date(
        data=stocks_income,
        date_col="date",
        target="stock_in",
        groupby_cols=["product_name", "date"],
        freq="D",
    )
    stocks_income = (
        stocks_income.groupby(["product_name", pd.Grouper(key="date", freq="M")])[
            "stock_in"
        ]
        .sum()
        .reset_index()
    )

    # create stock outcoming dataframe
    stocks_drop = stocks[["product_name", "date_end", "counts"]].rename(
        columns={"date_end": "date", "counts": "stock_out"}
    )
    stocks_drop = fill_skips_in_date(
        data=stocks_drop,
        date_col="date",
        target="stock_out",
        groupby_cols=["product_name", "date"],
        freq="D",
    )
    stocks_drop = (
        stocks_drop.groupby(["product_name", pd.Grouper(key="date", freq="M")])[
            "stock_out"
        ]
        .sum()
        .reset_index()
    )
    stocks_drop["stock_out"] = stocks_drop["stock_out"] * (-1)

    # combine income and outcome to calculate srocks for eact time point
    stocks = stocks_income.merge(stocks_drop, how="left").fillna(0)
    stocks["stocks"] = stocks["stock_in"] + stocks["stock_out"]
    stocks["stocks"] = stocks.groupby("product_name")["stocks"].cumsum()

    return stocks[["product_name", "date", "stocks"]]


def apply_preprocessing():
    """
    Применение всех шагов по предобработке данных.
    """

    # parse raw stocks
    data = parse_stocks()

    # preprocess parsed data
    data = preprocess_stocs(data=data)

    # create target
    data_agg = create_target_data(
        data=data,
        predicting_unit="product_name",
        date_col="date_start",
        target="counts",
        freq="D",
        groupby_cols=["product_name", "date_start"],
    )

    # create stocks data
    stocks = create_stocks(data=data)

    # decrease dtypes
    stocks = decrease_dtypes(data=stocks)
    data_agg = decrease_dtypes(data=data_agg)

    suffix = create_date_suffix()

    data_agg.to_parquet(
        os.path.join(
            modeling_paths.aggregated_data, f"target__aggregated_{suffix}.parquet"
        )
    )
    stocks.to_parquet(
        os.path.join(
            modeling_paths.aggregated_data, f"stocks__aggregated_{suffix}.parquet"
        )
    )

    print("\n[INFO] Data preprocessing is done.")


def main():
    """"""
    for path in [modeling_paths.data, modeling_paths.aggregated_data]:
        check_create_path(path=path)

    apply_preprocessing()


if __name__ == "__main__":
    main()
