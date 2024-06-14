"""
Точка входа в моделирование, в котором реализовано:

    1. предобработка данных
        - если данных нет признаков с суффиксом текущего периода - запускается предобработка и генерация признаков
    2. генерация признаков
    3. обучение
        - если в папке ./data/predictions/models нет 12(макс горизонт прогноза) моделей - запустится обучение
    4. прогноз
        - запускается без обучения в случае наличия моделей на каждый из 12 горизонтов прогнозирования

"""

import os

from modeling.conf import modeling_paths
from modeling.src.feature_engineering import main as generate_features
from modeling.src.predicting import main as forecasting
from modeling.src.preprocess_data import check_create_path, create_date_suffix
from modeling.src.preprocess_stocks import main as data_preprocessing
from modeling.src.training import main as training


def main():
    """

    in order to access .env variables run bash:
        export $(xargs <.env)

    """
    suffix = create_date_suffix()

    if not os.path.isdir(
        modeling_paths.features
    ) or f"features_{suffix}.parquet" not in os.listdir(modeling_paths.features):

        print(
            f"\n[INFO] No preprocessed data for current period ({suffix}): start preprocessing.\n"
        )

        data_preprocessing()

        generate_features()

    for path in [modeling_paths.predictions, modeling_paths.models]:
        check_create_path(path=path)

    n_trained_models = len(
        [i for i in os.listdir(modeling_paths.models) if "horizon" in i]
    )
    if n_trained_models != 12:
        print("\n[INFO] No trained models: start training.\n")
        training()
    else:
        forecasting()


if __name__ == "__main__":
    main()
