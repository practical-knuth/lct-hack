"""
Генерация признаков:

1. Трансформация циклических временных признаков
2. Макс длина подряд идущих нулей в окнах
3. Среднее значение 5 макс/мин значений
4. Роллинг агрегации
5. Лаговые значения
6. Признаки тренда
7. Окрестность прошлого года
8. Праздники
9. Обрезка каждого продукта по дате появления

"""

import datetime
import gc
import os
import shutil

import holidays
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modeling.conf import constants, features_params, modeling_params, modeling_paths
from modeling.src.preprocess_data import (
    check_create_path,
    create_date_suffix,
    decrease_dtypes,
)


def transform_cyclical_features(
    data: pd.DataFrame, date_col: str, agg_level: str
) -> pd.DataFrame:
    """
    Функция создает фичи на основе даты и производит sin/cos преобразование циклических признаков.
            - xsin = SIN((2 * pi * x) / max(x))
            - xcos = COS((2 * pi * x) / max(x))

    :param agg_level: str уровень аггрегации: [D, W, M]

    """
    available_features = ["weekday", "day", "dayofyear", "week", "month"]

    n_unique_values_map = {
        "weekday": 7,
        "day": 31,
        "dayofyear": 365,
        "week": 52,
        "month": 12,
    }

    assert np.isin(agg_level, ["D", "W", "M"]), "agg_level must be in ['D', 'W', 'M']"

    if agg_level == "D":
        data["day"] = data[date_col].dt.day
        data["dayofyear"] = data[date_col].dt.dayofyear
        data["weekday"] = data[date_col].dt.dayofweek
        data["week"] = data[date_col].dt.isocalendar().week
        data["month"] = data[date_col].dt.month

    elif agg_level == "W":
        data["week"] = data[date_col].dt.isocalendar().week
        data["month"] = data[date_col].dt.month

    else:
        data["month"] = data[date_col].dt.month

    cyclical_features = list(set(data.columns) & set(available_features))

    for feature in cyclical_features:
        data[feature] = data[feature].astype(int)

        data[f"{feature}_sin"] = np.sin(
            (data[feature] * 2 * np.pi) / n_unique_values_map[feature]
        )
        data[f"{feature}_cos"] = np.cos(
            (data[feature] * 2 * np.pi) / n_unique_values_map[feature]
        )

    data = data.drop(cyclical_features, axis=1)

    return decrease_dtypes(data=data)


def create_max_consecutive_zeros_feature(
    data: pd.DataFrame, target: str, gb_cols: list, windows: list
) -> pd.DataFrame:
    """
    Функция считает максимальное число подряд идущих нулей в заданном окне.

    :param data: pd.DataFrame с историей продаж
    :param target: str название целевой переменнов
    :param gb_cols: list колонки/а для группировки
    :param windows: list окон для расчета признака

    :return: pd.DataFrame
    """
    data.loc[data[target] == 0, "not_zero"] = 0
    data.loc[data[target] != 0, "not_zero"] = 1
    data["is_zero"] = 1 - data["not_zero"]

    try:
        data["consec_zeros"] = (
            data.groupby(gb_cols)
            .apply(
                lambda x: x["is_zero"]
                .groupby(x["not_zero"].cumsum())
                .transform(pd.Series.cumsum)
            )
            .values
        )
    # in case if we have 1 unique product_level we get ValueError
    except ValueError:
        data["consec_zeros"] = (
            data.groupby(gb_cols)
            .apply(
                lambda x: x["is_zero"]
                .groupby(x["not_zero"].cumsum())
                .transform(pd.Series.cumsum)
            )
            .values[0]
        )

    for window in windows:
        data[f"max_consec_zeros_w{window}_{target}"] = data.groupby(gb_cols)[
            "consec_zeros"
        ].transform(lambda x: x.rolling(window=window).max())

    data.drop(["not_zero", "is_zero", "consec_zeros"], axis=1, inplace=True)

    return data


def calculate_min_max_of_n_values(
    data: pd.DataFrame, target: str, gb_cols: list, stat: str, window: int
) -> pd.Series:
    """
    Функция расчитывает mean 5 max/min наблюдений в скользящем окне заданного размера.
    (Используется в create_roll_features)

    :param data: pd.DataFrame с историей продаж
    :param target: str название целевой переменной
    :param gb_cols: list колонок для группировки
    :param stat: str среднее 5-ти "max" or "min" значений в окне
    :param window: int размер окна

    :return: pd.Series с расчитанной статистикой
    """
    assert np.isin(stat, ["min", "max"]), "stat must be in ['min', 'max']"

    if stat == "max":
        mean_of_5_max_values = data.groupby(gb_cols)[target].transform(
            (
                lambda x: (x.rolling(window=window, min_periods=window)).apply(
                    lambda y: np.sort(y)[-5:].mean()
                )
            )
        )

        return mean_of_5_max_values
    else:
        mean_of_5_min_values = data.groupby(gb_cols)[target].transform(
            (
                lambda x: (x.rolling(window=window, min_periods=window)).apply(
                    lambda y: np.sort(y)[:5].mean()
                )
            )
        )

        return mean_of_5_min_values


def create_roll_features(
    data: pd.DataFrame,
    product_level: str,
    gb_cols: list,
    windows: list,
    target: str,
    agg_level: str,
) -> pd.DataFrame:
    """
    Функция создает признаки аггрегации по целевой переменной в скользящем окне.

    :param data: pd.DataFrame с историей продаж id_reg
    :param product_level: str колонка с прогнозируемой единицей
    :param gb_cols: list колонок для группировки
    :param windows: list размеров скользящего окна в зависимости от уровня аггрегации
    :param target: целевая переменная
    :param agg_level: str уровень аггрегации: [D, W, M]

    :return: pd.DataFrame с признаками
    """
    postfix = "_".join(gb_cols)

    # абсолютное изменение продаж (только елси группировка по прогнозируемой единице)
    if np.isin(product_level, gb_cols):
        data.loc[:, f"{target}_abs_change_{postfix}"] = data.groupby(gb_cols)[
            target
        ].transform(lambda x: np.abs(x - x.shift()))

    for window in windows:
        for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
            # квантили
            data.loc[
                :, f"{target}_roll_q{quantile}_w{window}_{postfix}"
            ] = data.groupby(gb_cols)[target].transform(
                lambda x: x.rolling(window=window, min_periods=window).quantile(
                    quantile
                )
            )
        # сумма квадратов
        data.loc[:, f"{target}_roll_squared_sum_w{window}_{postfix}"] = data.groupby(
            gb_cols
        )[target].transform(
            lambda x: np.square(x).rolling(window=window, min_periods=window).sum()
        )
        # доля не нулей
        data.loc[:, f"{target}_roll_nonzero_prop_w{window}_{postfix}"] = data.groupby(
            gb_cols
        )[target].transform(
            lambda x: (x != 0).rolling(window=window, min_periods=window).mean()
        )

        # фичи на основе abs_change (только елси группировка по прогнозируемой единице)
        if np.isin(product_level, gb_cols):
            # среднее абсолютное изменение продаж
            data.loc[:, f"{target}_mean_abs_change_w{window}_{postfix}"] = data.groupby(
                gb_cols
            )[f"{target}_abs_change_{postfix}"].transform(
                lambda x: x.rolling(window=window, min_periods=window).mean()
            )
            # std абсолютное изменение продаж
            data.loc[:, f"{target}_std_abs_change_w{window}_{postfix}"] = data.groupby(
                gb_cols
            )[f"{target}_abs_change_{postfix}"].transform(
                lambda x: x.rolling(window=window, min_periods=window).std()
            )

        # среднее 5 макс/мин значений

        # считаем только для окон >= 9, если aggLevel == W или M
        if np.isin(agg_level, ["W", "M"]):
            if window >= 9:
                # среднее значение 5 максимаьлных значений
                data.loc[:, f"{target}_mean_of_5max_w{window}_{postfix}"] = (
                    calculate_min_max_of_n_values(
                        data=data,
                        target=target,
                        gb_cols=gb_cols,
                        stat="max",
                        window=window,
                    )
                )
                # среднее значение 5 минимальных значений
                data.loc[:, f"{target}_mean_of_5min_w{window}_{postfix}"] = (
                    calculate_min_max_of_n_values(
                        data=data,
                        target=target,
                        gb_cols=gb_cols,
                        stat="min",
                        window=window,
                    )
                )
        # считаем все окна начианая с 14 если aggLevel == D
        else:
            if window >= 14:
                # среднее 5 максимальных значений в окне
                data.loc[:, f"{target}_mean_of_5max_w{window}_{postfix}"] = (
                    calculate_min_max_of_n_values(
                        data=data,
                        target=target,
                        gb_cols=gb_cols,
                        stat="max",
                        window=window,
                    )
                )
                # среднее 5 минимальных значений в окне
                data.loc[:, f"{target}_mean_of_5min_w{window}_{postfix}"] = (
                    calculate_min_max_of_n_values(
                        data=data,
                        target=target,
                        gb_cols=gb_cols,
                        stat="min",
                        window=window,
                    )
                )

    return data


def create_trend_features(
    data: pd.DataFrame,
    gb_cols: list,
    windows: list,
    target: str,
) -> pd.DataFrame:
    """
    Создание тренд-признаков:
        1. Сумма/Медиана таргета в window точках деленое на сумму/медиану таргета в предшествующих window точках
        2. Сумма/Медиана таргета в windows[0] точках на сумму/медиану таргета на каждую из windows[1:]

    :param data: pd.DataFrame с историей продаж id_reg
    :param gb_cols: list колонок для группировки
    :param windows: list размеров скользящего окна в зависимости от уровня аггрегации
    :param target: целевая переменная

    :return: pd.DataFrame с признаками
    """
    postfix = "_".join(gb_cols)

    # считаем фичи тренда
    for window in windows:
        for agg in ["sum", "median"]:
            first_window_col = f"{target}_rolling_w{windows[0]}_{agg}__{postfix}"
            new_col = f"{target}_rolling_w{window}_{agg}__{postfix}"

            data[new_col] = (
                data.groupby(gb_cols)[target].rolling(window=window).agg(agg).values
            )
            data[f"{new_col}_shift"] = data.groupby(gb_cols)[new_col].shift(window)

            data[f"{new_col}_trend"] = data[new_col] / data[f"{new_col}_shift"]

            # fix: y/0 -> 2 & 0/0 -> 1
            data.loc[
                data[[new_col, f"{new_col}_shift"]].sum(axis=1, skipna=False).eq(0),
                f"{new_col}_trend",
            ] = 1
            data.loc[data[f"{new_col}_trend"].eq(np.inf), f"{new_col}_trend"] = 2

            data = data.drop(f"{new_col}_shift", axis=1)

            if window != windows[0]:
                second_type_col = (
                    f"{target}_w{windows[0]}_{agg}/w{window}_{agg}__{postfix}"
                )

                data[second_type_col] = data[first_window_col] / data[new_col]

                # fix: y/0 -> 2 & 0/0 -> 1
                data.loc[
                    data[[first_window_col, new_col]].sum(axis=1, skipna=False).eq(0),
                    second_type_col,
                ] = 1
                data.loc[data[second_type_col].eq(np.inf), second_type_col] = 2

                data = data.drop(new_col, axis=1)

    # удаляем window sum фичи
    data = data.drop(
        [
            f"{target}_rolling_w{windows[0]}_sum__{postfix}",
            f"{target}_rolling_w{windows[0]}_median__{postfix}",
        ],
        axis=1,
    )

    return data


def create_last_year_locality_features(
    data: pd.DataFrame, global_id: str, target: str, agg_level: str, n: int
) -> pd.DataFrame:
    """
    Функция считает значения продаж в окрестности предыдущего года.

    :param data: pd.DataFrame с историе продаж
    :param global_id: str айди исследуемого разреза
    :param target: str целевая переменная
    :param agg_level: str уровень аггрегации: день/неделя/год
    :param n: int число точек до/после точки в прошлом году

    :return: pd.DataFrame с новыми признаками
    """
    assert np.isin(agg_level, ["D", "W", "M"]), "agg_level must be in ['D', 'W', 'M']"

    # карта размера годового цикла, в зависимости от уровня аггрегации
    cycle_size_map = {"M": 12, "W": 52, "D": 365}
    cycle = cycle_size_map[agg_level]

    # точки в окрестности прошлого года
    lags = range(cycle - n, cycle + n + 1)

    # считаем значения в окрестности прошлого года
    for lag in lags:
        data[f"{target}_y_ago_lag{lag}"] = data.groupby(global_id)[target].shift(lag)

    return data


def create_lag_features(
    data: pd.DataFrame, gb_cols: list, min_lag: int, max_lag: int, target: str
) -> pd.DataFrame:
    """
    Функция считает лаг признаки в диапазоне min_lag - max_lag
    и значения lag/lag+1.

    -- Если значение лага > 0, а значение лага+1 == 0, при делении первого на
    второй будет inf, такие значения меняем на 2, предполагаем увеличение на 100% --

    :param data: pd.DataFrame с историей продаж
    :param gb_cols: list столбцы для группировки
    :param target: str название колонки, по которой создаются признаки
    :param min_lag: int значение минимального лага
    :param max_lag: int значение максимального лага

    :return: pd.DataFrame с новыми признаками
    """
    # считаем лаги
    for lag in range(min_lag, max_lag + 1):
        data[f"{target}_lag{lag}"] = data.groupby(gb_cols)[target].shift(lag)

        if lag < max_lag:
            data[f"{target}_lag{lag}/lag{lag+1}"] = data.groupby(gb_cols)[target].shift(
                lag
            ) / data.groupby(gb_cols)[target].shift(lag + 1)

    # исправляем inf (вызванные y/0, где y > 0) и лишние NaN (вызванные 0/0)
    for lag in range(min_lag, max_lag):
        # если оба значения == 0, в {target}_lag{lag}/lag{lag+1} будет NaN, поэтому заполняем их единицей
        data.loc[
            data[[f"{target}_lag{lag}", f"{target}_lag{lag+1}"]]
            .sum(axis=1, skipna=False)
            .eq(0),
            f"{target}_lag{lag}/lag{lag+1}",
        ] = 1

        # если числитель > 0 и знаменатель == 0, в {target}_lag{lag}/lag{lag+1} inf меняем на 2
        data.loc[
            data[f"{target}_lag{lag}/lag{lag+1}"].eq(np.inf),
            f"{target}_lag{lag}/lag{lag+1}",
        ] = 2

    return data


def calculate_nearest_holidays(
    date, holidays_data: pd.DataFrame, date_col: str, agg_level: str, future=True
) -> pd.DataFrame:
    """
    Функция счтает количество периодов до ближайшего праздника/нового года.

    :param date: pd.datetime дата
    :param holidays_data: pd.DataFrame с датами праздников в России
    :param date_col: str название колонки с датой в holidays_data
    :param agg_level: str уровень аггрегации day/week/month
    :param future: bool - True(default) - до ближайшего праздника в будущем
                        - False - до ближайшего празднкика в прошлом

    :return: pd.Series с дистанциями до ближайшего праздника
    """
    # делитель количества дней в зависимости от уровня аггрегации
    days_in_agg_level_map = {"W": 7, "D": 1, "M": 30}
    divider = days_in_agg_level_map[agg_level]

    # дней до всех праздников
    days_to_holiday = (date - holidays_data[date_col]).dt.days

    # периодов до ближайшего праздника
    periods_to_hol = (
        np.min(np.abs(days_to_holiday[(days_to_holiday == 0) | (days_to_holiday < 0)]))
        / divider
    )
    if not future:
        periods_to_hol = (
            np.min(
                np.abs(days_to_holiday[(days_to_holiday == 0) | (days_to_holiday > 0)])
            )
            / divider
        )

    return periods_to_hol


def create_holidays_features(
    data: pd.DataFrame, name_holiday: str, date_col: str, agg_level: str
) -> pd.DataFrame:
    """
    Функция создает признаки на основе праздников:
        - дистанция до ближайшего праздника в будущем/в прошлом
        - дистанция до нового года в будущем/в прошлом

    :param data: pd.DataFrame с истрией продаж
    :param holidays_data: pd.DataFrame с датами праздников
    :param date_col: str - название колонки с датой
    :param name_holiday: str - название праздника
    :param agg_level: str уровень аггрегации day/week/month

    :return: pd.DataFrame с новыми признаками holidays
    """
    data["year"] = data[date_col].dt.year
    # cоздаем датасет праздников
    years_list = list(range(int(min(data["year"])) - 1, int(max(data["year"])) + 2))
    holiday_list = []
    for holiday in holidays.Russia(years=years_list).items():
        holiday_list.append(holiday)

    # Все праздники
    holidays_df = pd.DataFrame(holiday_list, columns=[date_col, "holiday"])

    if name_holiday == "ny":
        # Празники нового года
        holidays_df = holidays_df[holidays_df["holiday"] == "New Year Holidays"]

    # date -> pd.to_datetime
    holidays_df[date_col] = pd.to_datetime(holidays_df[date_col])

    # сортируем по дате
    holidays_df = holidays_df.sort_values(by=date_col)

    # отложим только уникальные даты
    unique_dates = data[[date_col]].drop_duplicates()

    # признаки до праздников
    unique_dates[f"{agg_level}_to_{name_holiday}_future"] = unique_dates[
        date_col
    ].transform(
        lambda x: calculate_nearest_holidays(
            date=x, holidays_data=holidays_df, date_col=date_col, agg_level=agg_level
        )
    )

    # признаки после праздников
    unique_dates[f"{agg_level}_to_{name_holiday}_past"] = unique_dates[
        date_col
    ].transform(
        lambda x: calculate_nearest_holidays(
            date=x,
            holidays_data=holidays_df,
            date_col=date_col,
            agg_level=agg_level,
            future=False,
        )
    )

    # соединяем признаки к основном датафрейму
    data = data.merge(unique_dates, how="left")

    return data


def create_targets(
    data: pd.DataFrame, horizont: int, targets: list, predicting_unit: str
) -> pd.DataFrame:
    """
    Создание таргетов в зависимости от горизонта прогнозирования.

    :param data: pd.DataFrame с фичами
    :param horizont: int горизонт прогноза
    :param targets: list сырых таргетов
    :param predicting_unit: str колонка с идентификатором прогнозируемой единицы

    :return: pd.DataFrame
    """
    df = data.copy()

    del data
    gc.collect()

    generated_targets = []

    for horizont in range(1, horizont + 1):
        for target in targets:
            if horizont < 10:
                current_target = f"{target}_target_0{horizont}"
                df[current_target] = df.groupby([predicting_unit])[target].shift(
                    -horizont
                )
            else:
                current_target = f"{target}_target_{horizont}"
                df[current_target] = df.groupby([predicting_unit])[target].shift(
                    -horizont
                )

            generated_targets.append(current_target)

    # drop original targets
    df = df.drop(targets, axis=1)

    return df


def cut_first_sale(
    data: pd.DataFrame, predicting_unit: str, targets: list, date_col: str
) -> pd.DataFrame:
    """"""
    first_sale = (
        data.groupby(predicting_unit)
        .apply(lambda x: x[(x[targets] > 0).any(axis=1).values][date_col].min())
        .reset_index()
        .rename(columns={0: "first_sale"})
    )

    data = data.merge(first_sale, on=predicting_unit)
    data = data[data[date_col] >= data["first_sale"]].reset_index(drop=True)

    return data.drop("first_sale", axis=1)


def apply_feature_engineering(
    data: pd.DataFrame, tmp_features_folder: str
) -> pd.DataFrame:
    """
    All feature engineering functions applying.

    :param data: pd.DataFrame агрегированные данные с историей целевой переменной
    :param tmp_features_folder: str путь к временной папке для сохранения кусков признаков,
                                    чтобы не держать их в памяти

    :return: None
    """
    suffix = create_date_suffix()
    targets = features_params.targets

    # date based data columns
    date_data = data[[constants.date]].drop_duplicates()

    # keep only fe required columns
    data_all = data[[constants.predicting_unit, constants.date] + targets]

    # 1. targets
    classic_targets = create_targets(
        data=data_all.copy(),
        horizont=modeling_params.horizon,
        targets=features_params.targets,
        predicting_unit=constants.predicting_unit,
    )
    classic_targets.to_parquet(
        os.path.join(modeling_paths.targets, f"targets_{suffix}.parquet")
    )

    del classic_targets
    gc.collect()

    # 2. max consecutive zeros

    if len(features_params.consecutive_windows) > 0:
        max_consec_zeros_dfs = Parallel(n_jobs=features_params.n_jobs)(
            delayed(create_max_consecutive_zeros_feature)(
                data=data_all.copy(),
                target=target,
                gb_cols=[constants.predicting_unit],
                windows=features_params.consecutive_windows,
            )
            for target in targets
        )

        max_consec_zeros_df = pd.concat(max_consec_zeros_dfs, axis=1)
        max_consec_zeros_df_final = max_consec_zeros_df.loc[
            :, ~max_consec_zeros_df.columns.duplicated()
        ]

        # save to temporary folder
        max_consec_zeros_df_final.to_parquet(
            os.path.join(tmp_features_folder, "max_consecutive_zeros.parquet")
        )

        del max_consec_zeros_dfs[:], max_consec_zeros_df, max_consec_zeros_df_final
        gc.collect()

    # 3. previous year locality

    if features_params.previous_year_locality_n > 0:
        prev_year_locality_dfs = Parallel(n_jobs=features_params.n_jobs)(
            delayed(create_last_year_locality_features)(
                data=data_all.copy(),
                target=target,
                global_id=constants.predicting_unit,
                agg_level=constants.freq,
                n=features_params.previous_year_locality_n,
            )
            for target in targets
        )

        prev_year_locality_df = pd.concat(prev_year_locality_dfs, axis=1)
        prev_year_locality_df_final = prev_year_locality_df.loc[
            :, ~prev_year_locality_df.columns.duplicated()
        ].copy()

        # save to temporary folder
        prev_year_locality_df_final.to_parquet(
            os.path.join(tmp_features_folder, "prev_year_locality.parquet")
        )

        del (
            prev_year_locality_dfs[:],
            prev_year_locality_df,
            prev_year_locality_df_final,
        )
        gc.collect()

    # 4. lag features

    if features_params.max_lag > 0:
        lag_features_dfs = Parallel(n_jobs=features_params.n_jobs)(
            delayed(create_lag_features)(
                data=data_all.copy(),
                target=target,
                gb_cols=[constants.predicting_unit],
                min_lag=features_params.min_lag,
                max_lag=features_params.max_lag,
            )
            for target in targets
        )

        lag_features_df = pd.concat(lag_features_dfs, axis=1)
        lag_features_df_final = lag_features_df.loc[
            :, ~lag_features_df.columns.duplicated()
        ].copy()

        # save to temporary folder
        lag_features_df_final.to_parquet(
            os.path.join(tmp_features_folder, "lags.parquet")
        )

        del lag_features_dfs[:], lag_features_df, lag_features_df_final
        gc.collect()

    # 5. rolling aggregations (groupby predicting_unit)

    if len(features_params.rolling_windows) > 0:
        roll_features_dfs = Parallel(n_jobs=features_params.n_jobs)(
            delayed(create_roll_features)(
                data=data_all.copy(),
                product_level=constants.predicting_unit,
                target=target,
                gb_cols=[constants.predicting_unit],
                windows=features_params.rolling_windows,
                agg_level=constants.freq,
            )
            for target in targets
        )

        roll_features_df = pd.concat(roll_features_dfs, axis=1)
        roll_features_df_final = roll_features_df.loc[
            :, ~roll_features_df.columns.duplicated()
        ].copy()

        # save to temporary folder
        roll_features_df_final.to_parquet(
            os.path.join(tmp_features_folder, "rollings.parquet")
        )

        del roll_features_dfs[:], roll_features_df, roll_features_df_final
        gc.collect()

    # 6. trend features

    if len(features_params.trend_windows) > 0:

        trend_features_dfs = Parallel(n_jobs=features_params.n_jobs)(
            delayed(create_trend_features)(
                data=data_all.copy(),
                gb_cols=[constants.predicting_unit],
                windows=features_params.trend_windows,
                target=target,
            )
            for target in targets
        )

        trend_features_df = pd.concat(trend_features_dfs, axis=1)
        trend_features_df_final = trend_features_df.loc[
            :, ~trend_features_df.columns.duplicated()
        ].copy()

        # save to temporary folder
        trend_features_df_final.to_parquet(
            os.path.join(tmp_features_folder, "trends.parquet")
        )

        del trend_features_dfs[:], trend_features_df, trend_features_df_final
        gc.collect()

    # 7. cyclical features transformation

    if features_params.cyclical:
        cyclical_features_df = transform_cyclical_features(
            data=date_data,
            date_col=constants.date,
            agg_level=constants.freq,
        )

    # 8. holidays

    if features_params.holidays:
        # all holidays
        holidays_df = create_holidays_features(
            data=data_all[[constants.date]].drop_duplicates().copy(),
            date_col=constants.date,
            name_holiday="holidays",
            agg_level=constants.freq,
        )

        # ny holidays
        new_year_df = create_holidays_features(
            data=data_all[[constants.date]].drop_duplicates().copy(),
            date_col=constants.date,
            name_holiday="ny",
            agg_level=constants.freq,
        )
        holidays_df = holidays_df.merge(new_year_df)

        del new_year_df
        gc.collect()

    # merge all features

    # cyclical features & holdays

    if features_params.cyclical:
        data_all = data_all.merge(cyclical_features_df, on=constants.date)
    if features_params.holidays:
        data_all = data_all.merge(holidays_df, on=constants.date)

    # target based features
    order_cols = [constants.predicting_unit, constants.date]

    data_all = data_all.sort_values(by=order_cols).reset_index(drop=True)

    for file in os.listdir(tmp_features_folder):
        if ".parquet" in file:
            features = pd.read_parquet(os.path.join(tmp_features_folder, file))
            if (
                np.mean(
                    data_all.reset_index(drop=True)[order_cols]
                    == features[order_cols].reset_index(drop=True)[order_cols]
                )
                != 1
            ):
                features = features.sort_values(by=[order_cols])

            features_columns = features.columns[
                ~features.columns.isin(data_all.columns)
            ]

            data_all = pd.concat([data_all, features[features_columns]], axis=1)

            del features
            gc.collect()

    data_all = decrease_dtypes(data=data_all)

    data_all = cut_first_sale(
        data=data_all,
        targets=targets,
        date_col=constants.date,
        predicting_unit=constants.predicting_unit,
    )

    data_all.to_parquet(
        os.path.join(modeling_paths.features, f"features_{suffix}.parquet")
    )

    del data_all
    gc.collect()


def main():
    """"""
    suffix = create_date_suffix()
    start = datetime.datetime.now()

    tmp_features_folder = os.path.join(modeling_paths.features, "tmp")

    for path in [modeling_paths.targets, modeling_paths.features, tmp_features_folder]:
        check_create_path(path=path)

    # main data
    data_all = pd.read_parquet(
        os.path.join(
            modeling_paths.aggregated_data, f"target__aggregated_{suffix}.parquet"
        )
    )

    apply_feature_engineering(data=data_all, tmp_features_folder=tmp_features_folder)

    shutil.rmtree(tmp_features_folder)

    end = datetime.datetime.now()

    print(f"\n[INFO] Feature Engineering: elapsed time is {(end - start) / 60}")


if __name__ == "__main__":
    main()
