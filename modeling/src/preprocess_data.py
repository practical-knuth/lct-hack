import datetime
import gc
import os

import numpy as np
import pandas as pd

from modeling.conf import constants, modeling_paths, preprocessing_constants, raw_paths


def create_date_suffix() -> str:
    """
    Создание суффикса из текущей даты.
    """
    today = datetime.datetime.now().date()
    suffix = "".join([str(today.year), str(today.month), str(today.day)])

    return suffix


def check_create_path(path: str):
    """
    Проверка наличия и создание папки если ее нет.

    :param path: str путь к папке
    """
    if not os.path.exists(path):
        os.mkdir(path)


def rename_columns(
    data: pd.DataFrame, new_columns_map: dict, data_cat: str
) -> pd.DataFrame:
    """
    Переименовывание колонок

    :param data: pd.DataFrame для трансформации
    :param new_columns_map: dict старых и новых наименований колонок
    :param data_cat: str категория текущих данных

    :return: pd.DataFrame
    """
    data = data.rename(columns=new_columns_map)
    data = data[list(new_columns_map.values())]
    data["type"] = data_cat

    return data


def preprocess_strings(data: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка str колонок:
        - strip & lower

    :param data: pd.DataFrame для трансформации

    :return: pd.DataFrame
    """
    str_columns = data.columns[data.dtypes == "object"]

    for col in str_columns:
        data[col] = data[col].str.strip().str.lower()

    return data


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Удаление дупликатов

    :param data: pd.DataFrame для трансформации

    :return: pd.DataFrame
    """
    duplicated_idxs = data[data.drop(["type"], axis=1).duplicated()].index

    return data.drop(duplicated_idxs).reset_index(drop=True)


def drop_nans(data: pd.DataFrame, checking_columns: list) -> pd.DataFrame:
    """
    Удаление пропусков

    :param data: pd.DataFrame для трансформации
    :param checking_columns: list колонок для удаления пропусков

    :return: pd.DataFrame
    """
    drop_index = data[data[checking_columns].isna().any(axis=1)].index

    return data.drop(drop_index)


def drop_inf(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Удаление пропусков

    :param data: pd.DataFrame для трансформации
    :param target: str колонка с таргетом

    :return: pd.DataFrame
    """
    drop_index = data[data[target].eq(np.inf)].index

    return data.drop(drop_index)


def transform_to_datetime(data: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """
    Приведение колонки с датой к datetime формату

    :param data: pd.DataFrame для трансформации
    :param date_columns: list колонок с датой

    :return: pd.DataFrame
    """
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])

    return data


def decrease_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """dtypes decreasing"""

    data[data.columns[data.dtypes == "int64"]] = data[
        data.columns[data.dtypes == "int64"]
    ].astype("int32")

    data[data.columns[data.dtypes == "float64"]] = data[
        data.columns[data.dtypes == "float64"]
    ].astype("float32")

    return data


def fill_skips_in_date(
    data: pd.DataFrame, date_col: str, groupby_cols: list, target: str, freq: str = "ME"
) -> pd.DataFrame:
    """
    Функция заполняет пропуски в дате нулями.

    :param data: pd.DataFrame с историей пасспотока
    :param date_col: str колонка с датой
    :param groupby_cols: list столбцов для группировки
    :param target: str целевых переменных
    :param region: int номер региона
    :param freq: str='M' гранулярность

    :return: pd.DataFrame с заполненными пропусками в датах
    """
    assert np.isin(freq, ["D", "W", "ME"]), "freq must be in ['D', 'W', 'M']"

    df = data.copy()

    # create all dates from min to max
    all_dates = pd.DataFrame(
        {date_col: pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)}
    )

    # left merge all dates to data_raw
    all_dates_data = all_dates.merge(df, how="left")

    # define features
    features = list(set(all_dates_data.columns) - set([target]))

    # fill features with ffill & bfill
    all_dates_data[features] = all_dates_data[features].ffill().bfill()

    # fill target with 0
    all_dates_data[target] = all_dates_data[target].fillna(0)

    # unstack, stack to create all dates dataframe
    all_dates_data = (
        all_dates_data.groupby(groupby_cols)[target]
        .sum()
        .unstack(fill_value=0)
        .stack()
        .reset_index()
        .rename(columns={0: target})
    )

    return all_dates_data


def apply_preprocessing():
    """
    Применение всех этапов предобработки данных.
    """

    # read data
    data_s = pd.read_parquet(raw_paths.spgz_selected)
    data_k = pd.read_parquet(raw_paths.kpgz_selected)

    preprocessed_data = []

    for d, new_cols, cat in zip(
        [data_s, data_k],
        [
            preprocessing_constants.rename_columns_s,
            preprocessing_constants.rename_columns_k,
        ],
        ["spgz", "kpgz"],
    ):
        # rename and select columns
        d = rename_columns(data=d, new_columns_map=new_cols, data_cat=cat)

        preprocessed_data.append(d)

    data = pd.concat(preprocessed_data)

    del preprocessed_data[:]
    gc.collect()

    # decrease dtypes
    data = decrease_dtypes(data=data)

    # preprocess string columns
    data = preprocess_strings(data)

    # drop diplicates
    data = drop_duplicates(data=data)

    # drop nans
    data = drop_nans(
        data=data,
        checking_columns=[constants.target, constants.date, constants.predicting_unit],
    )

    # drop inf values
    data = drop_inf(data=data, target=constants.target)

    # transform date column
    data = transform_to_datetime(data=data, date_columns=[constants.date])

    # fill skips in date
    data_agg = fill_skips_in_date(
        data=data,
        date_col=constants.date,
        target=constants.target,
        groupby_cols=["type", constants.predicting_unit, constants.date],
        freq=constants.freq,
    )

    suffix = create_date_suffix()

    data_agg.to_parquet(
        os.path.join(
            modeling_paths.aggregated_data, f"target__aggregated_{suffix}.parquet"
        )
    )


def main():
    """"""
    for path in [modeling_paths.data, modeling_paths.aggregated_data]:
        check_create_path(path=path)

    apply_preprocessing()


if __name__ == "__main__":
    main()
