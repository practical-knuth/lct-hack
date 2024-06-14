"""
Прогнозирование спроса с применением предобученных моделей.

Применяемы шаги:

    1. Загружается обученная модель и делается прогноз на следующие 12 месяцев
    2. Если модели нет, запускается обучение и последующее сохраниение модели
    
"""

import datetime
import os

import pandas as pd
from catboost import CatBoostError, CatBoostRegressor
from tqdm import tqdm

from modeling.conf import constants, modeling_params, modeling_paths
from modeling.src.preprocess_data import check_create_path, create_date_suffix
from modeling.src.training import make_prediction, postprocess_prediction, train_model
from modeling.src.utils_training import train_val_split


def run(
    target: str,
    features_path: str,
    targets_path: str,
    save_path: str,
) -> pd.DataFrame:
    """
    Запуск прогноза.

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

        try:
            model = CatBoostRegressor()
            model.load_model(os.path.join(horizon_folder, "model.cbm"))

            selected_features = model.feature_names_
        except CatBoostError:
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

    print(f"\n[INFO] Forecasting: elapsed time is {(end - start) / 60}")


if __name__ == "__main__":
    main()
