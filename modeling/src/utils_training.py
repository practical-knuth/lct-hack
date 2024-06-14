import gc

import numpy as np
import pandas as pd
from catboost import Pool


def train_val_split(
    data: pd.DataFrame,
    target: str,
    val_rate: float,
    date_col: str,
    product_level: str,
    freq: str = "M",
) -> tuple:
    """
    Функция разделяет данные на трейн/вал-тест по дате.

    :param data: pd.DataFrame с данными для разделения
    :param target: str колонка с таргетом
    :param val_rate: float доля данных для валидационного сэта
    :param date_col: str название колонки с датой
    :param product_level: str колонка уровня агрегации по товарам

    return [0] - тренировочная выборка
           [1] - вал выборка
           [2] - выборка будущего периода
    """
    train = data[~data[target].isna()].reset_index(drop=True)
    next_period_data = data[data[target].isna()].reset_index(drop=True)

    all_dates = pd.date_range(train[date_col].min(), train[date_col].max(), freq=freq)

    val_start = all_dates[-int(np.round(len(all_dates) * val_rate))]

    val = train[train[date_col] >= val_start]
    train = train[train[date_col] < val_start]

    return train, val, next_period_data


def create_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    features: list,
    cat_features: list = [],
):
    """
    Функция создает train/val датасеты для Catboost.

    :param model: str название модели, должно быть LGB или Catboost
    :param train: pd.DataFrame тренировочная выборка
    :param y_train: pd.Series тренировоччная целевая переменная
    :param val: pd.DataFrame валидационная выборка
    :param y_val: pd.Series валиационная целевая переменная
    :param features: list признаков
    :param cat_features: list категориальных признаков

    :return: [0] - тренировочный датасет
             [1] - валидационный датасет
    """
    train.loc[:, cat_features] = train[cat_features].astype(str)
    val.loc[:, cat_features] = val[cat_features].astype(str)

    # индексы категориальных признаков
    cat_cols_idxs = train.columns.get_indexer(cat_features)

    train_dataset = Pool(
        data=np.array(train),
        label=np.array(y_train),
        feature_names=features,
        cat_features=cat_cols_idxs,
    )
    val_dataset = Pool(
        data=np.array(val),
        label=np.array(y_val),
        feature_names=features,
        cat_features=cat_cols_idxs,
    )

    del (
        train,
        val,
        cat_cols_idxs,
    )
    gc.collect()

    return train_dataset, val_dataset
