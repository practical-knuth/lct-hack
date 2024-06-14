import os
from dataclasses import dataclass

from .constants import constants


@dataclass
class Paths:
    targets: str
    references: str
    stocks: str
    spgz_selected: str
    kpgz_selected: str


@dataclass
class ModelingPaths:
    data: str
    aggregated_data: str
    targets: str
    features: str
    predictions: str
    models: str


raw_paths = Paths(
    targets="./data/target_fetched",
    references="./data/КПГЗ ,СПГЗ, СТЕ.xlsx",
    stocks="./data/Складские_остатки/",
    spgz_selected="./data/target_fetched/target_spqz.parquet",
    kpgz_selected="./data/target_fetched/target_kpgz.parquet",
)

modeling_paths = ModelingPaths(
    data="./modeling//data/",
    aggregated_data="./modeling/data/aggregated/",
    targets="./modeling/data/targets/",
    features="./modeling/data/features/",
    predictions="./modeling//data/predictions/",
    models="./modeling/data/predictions/models/",
)
