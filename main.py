import os

from modeling.modeling import main as training
from modeling.src.preprocess_data import create_date_suffix
from tg_bot.constants import constants
from tg_bot.src.tg_bot import run_bot


def main():
    suffix = create_date_suffix()
    prediction_name = f"predictions_{suffix}.parquet"

    prediction_path = os.path.join(constants.prediction_path, prediction_name)
    if not os.path.exists(prediction_path):
        training()

    run_bot()


if __name__ == "__main__":

    main()
