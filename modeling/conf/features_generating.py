from dataclasses import dataclass

from .constants import constants


@dataclass
class FeaturesGenerating:
    targets: list
    n_jobs: int
    consecutive_windows: list
    previous_year_locality_n: int
    max_lag: int
    min_lag: int
    rolling_windows: list
    trend_windows: list
    cyclical: bool
    holidays: bool


features_params = FeaturesGenerating(
    targets=[constants.target],
    n_jobs=1,
    consecutive_windows=[6, 12],
    previous_year_locality_n=3,
    max_lag=6,
    min_lag=0,
    rolling_windows=[3, 6, 12],
    trend_windows=[3, 6, 12],
    cyclical=True,
    holidays=True,
)
