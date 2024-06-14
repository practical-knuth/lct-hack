"""
Обучение catboost с отбором признаков на основе Shap values.

Применяемы шаги:

    1. Прогоняется отбор признаков до минимального числа
    2. Находится точка (число фичей) с наилучшим лоссом
    3. Проводится отбор признаков до числа из 2 пункта
    
"""

import datetime
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, EFeaturesSelectionAlgorithm, EShapCalcType, Pool
from tqdm import tqdm

from modeling.conf import constants, modeling_params, modeling_paths
from modeling.src.preprocess_data import check_create_path, create_date_suffix
from modeling.src.utils_training import create_datasets, train_val_split


def find_best_features(
    model: CatBoostRegressor,
    train_pool: Pool,
    eval_pool: Pool,
    steps: int,
    initial_features_number: int,
) -> list:
    """
    Поиск лучшего набора фичей:
        1. Отбор минимального числа фичей;
        2. Поиск оптимальной точки по кривой лосса.

    :param model: CatBoostRegressor обученная модель
    :param train_pool: catboost.Pool тренировочный сет
    :param eval_pool: catboost.Pool валидационный сет
    :param num_features_to_select: int число отбираемых фичей на первом шаге
    :param initial_features_number: int изначальное число фичей

    :return: array
    """
    # param for catboost feature_selection (range represented as a string)
    features_for_select = f"0-{initial_features_number - 1}"

    #  first run of features selection
    summary = model.select_features(
        train_pool,
        eval_set=eval_pool,
        features_for_select=features_for_select,
        num_features_to_select=10,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        steps=steps,
        train_final_model=False,
        plot=False,
        logging_level="Silent",
    )

    # best loss index
    best_iter = np.argmin(summary["loss_graph"]["loss_values"])
    # number of selected features of best loss
    n_features_to_select = (
        initial_features_number
        - summary["loss_graph"]["removed_features_count"][best_iter]
    )

    # final features selection run
    summary = model.select_features(
        train_pool,
        eval_set=eval_pool,
        features_for_select=features_for_select,
        num_features_to_select=n_features_to_select,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        steps=steps,
        train_final_model=False,
        plot=False,
        logging_level="Silent",
    )

    return summary["selected_features_names"]


def train_model(
    train: pd.DataFrame, val: pd.DataFrame, features: list, target: str
) -> tuple:
    """
    Обучение моделей

    :param train: pd.DataFrame обучающая выборка
    :param val: pd.DataFrame валидац. выборка
    :param features: list обучающих признаков
    :param target: str колонка с таргетом

    :return: tuple
    """
    # create train/val pools
    train_pool, val_pool = create_datasets(
        train=train[features],
        y_train=train[target],
        val=val[features],
        y_val=val[target],
        features=features,
    )

    # train model
    model = CatBoostRegressor(**modeling_params.cb_params)
    model.fit(train_pool, eval_set=val_pool)

    # select features
    selected_features = find_best_features(
        model=model,
        train_pool=train_pool,
        eval_pool=val_pool,
        steps=modeling_params.feature_selection_steps_n,
        initial_features_number=len(features),
    )

    # retrain final model
    full_train = pd.concat([train, val]).reset_index(drop=True)

    # update params
    new_params = model.get_params()
    del (
        new_params["early_stopping_rounds"],
        new_params["best_model_min_trees"],
        new_params["use_best_model"],
    )
    new_params["iterations"] = int(new_params["iterations"] * modeling_params.val_rate)

    model = CatBoostRegressor(**new_params)

    model.fit(full_train[selected_features], full_train[target])

    return model, selected_features


def make_prediction(
    data: pd.DataFrame, features: list, predicting_unit: str, date_col: str, model
) -> pd.DataFrame:
    """
    Прогнозирование

    :param data: pd.DataFrame обучающая выборка
    :param features: list обучающих признаков
    :param predicting_unit: str колонка с прогнозируемой единицей
    :param date_col: str колонка с датой
    :param model: catboost.core.CatBoostRegressor обученная модель

    :return: pd.DataFrame
    """
    prediction = model.predict(data[features])

    next_period_prediction = pd.DataFrame(
        {
            predicting_unit: data[predicting_unit],
            date_col: data[date_col],
            "prediction": prediction,
        }
    )

    next_period_prediction = (
        next_period_prediction.groupby(predicting_unit)
        .apply(lambda x: x[x[date_col] == x[date_col].max()])
        .reset_index(drop=True)
    )

    return next_period_prediction


def postprocess_prediction(data: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    """
    Постпрогнозная обработка:
        - зануление отрицательных значений
        - округление

    :param data: pd.DataFrame c прогнозами
    :param prediction_col: str колонка с прогнозами

    :returm: pd.DataFrame
    """
    data.loc[data[prediction_col] < 0, prediction_col] = 0
    data[prediction_col] = np.round(data[prediction_col])

    return data


def run(
    target: str,
    features_path: str,
    targets_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Запуск обучения и прогноза

    :param target: str прогнозируемый таргет
    :param features_path: str путь к призанкам
    :param targets_path: str путь к таргетам
    :param save_path: str путь для сохранения прогнозов

    :return: pd.DataFrame
    """
    suffix = create_date_suffix()
    data = pd.read_parquet(os.path.join(features_path, f"features_{suffix}.parquet"))
    targets = pd.read_parquet(os.path.join(targets_path, f"targets_{suffix}.parquet"))

    target_columns = list(
        set(targets.columns) - set([constants.predicting_unit, constants.date])
    )

    data = data.merge(targets)

    features = list(
        set(data.columns) - set(modeling_params.drop_features) - set(target_columns)
    )

    # массив для прогнозов
    all_preds_storage = []

    for horizon in tqdm(range(1, modeling_params.horizon + 1)):
        horizon_folder = os.path.join(modeling_paths.models, f"horizon_{horizon}")

        check_create_path(path=horizon_folder)

        current_data = data.copy()

        date_offset_param = {"months": horizon}

        if horizon < 10:
            current_target_col = f"{target}_target_0{horizon}"
        else:
            current_target_col = f"{target}_target_{horizon}"

        # add periods to dates
        current_data[constants.date] = current_data[constants.date] + pd.DateOffset(
            **date_offset_param
        )

        # train/val//next_period split
        train, val, next_period = train_val_split(
            data=current_data,
            target=current_target_col,
            val_rate=modeling_params.val_rate,
            date_col=constants.date,
            product_level=constants.predicting_unit,
            freq=constants.freq,
        )

        # train model
        model, selected_features = train_model(
            train=train, val=val, features=features, target=current_target_col
        )
        model.save_model(os.path.join(horizon_folder, "model.cbm"), format="cbm")

        next_period_prediction = make_prediction(
            data=next_period,
            features=selected_features,
            predicting_unit=constants.predicting_unit,
            date_col=constants.date,
            model=model,
        )

        all_preds_storage.append(next_period_prediction)

    # save results
    all_preds = pd.concat(all_preds_storage)
    # postprocess
    all_preds = postprocess_prediction(data=all_preds, prediction_col="prediction")

    suffix = create_date_suffix()

    all_preds.to_parquet(os.path.join(save_path, f"predictions_{suffix}.parquet"))


def main():
    """"""
    start = datetime.datetime.now()

    # disable SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    for path in [modeling_paths.predictions, modeling_paths.models]:
        check_create_path(path=path)

    run(
        target=modeling_params.target,
        features_path=modeling_paths.features,
        targets_path=modeling_paths.targets,
        save_path=modeling_paths.predictions,
    )

    end = datetime.datetime.now()

    print(f"\n[INFO] Training: elapsed time is {(end - start) / 60}")


if __name__ == "__main__":
    main()
