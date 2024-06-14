from dataclasses import dataclass

from .constants import constants


@dataclass
class Modeling:
    target: str
    horizon: int
    drop_features: list
    val_rate: float
    cb_params: dict
    feature_selection_steps_n: int


modeling_params = Modeling(
    target=constants.target,
    horizon=12,
    drop_features=[constants.target, constants.date, constants.predicting_unit],
    val_rate=0.2,
    cb_params={
        "iterations": 50,
        "max_depth": 4,
        "early_stopping_rounds": 50,
        "best_model_min_trees": 50,
        "use_best_model": True,
        "random_seed": 42,
        "loss_function": "RMSE",
        "task_type": "CPU",
        "verbose": 0,
    },
    feature_selection_steps_n=10,
)
