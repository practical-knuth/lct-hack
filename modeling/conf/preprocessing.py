from dataclasses import dataclass

from .constants import constants


@dataclass
class DataPreprocessing:
    rename_columns_s: dict
    rename_columns_k: dict


preprocessing_constants = DataPreprocessing(
    rename_columns_s={
        "Наименование СПГЗ": constants.predicting_unit,
        "Дата регистрации": constants.date,
        "Цена ГК, руб.": constants.target,
    },
    rename_columns_k={
        "Конечное наименование КПГЗ": constants.predicting_unit,
        "Дата регистрации": constants.date,
        "Цена ГК, руб.": constants.target,
    },
)
