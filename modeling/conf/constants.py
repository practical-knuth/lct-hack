from dataclasses import dataclass


@dataclass
class Constants:
    predicting_unit: str
    date: str
    target: list
    freq: str


constants = Constants(
    predicting_unit="product_name", date="date", target="counts", freq="M"
)
