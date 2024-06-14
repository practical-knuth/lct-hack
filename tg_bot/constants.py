from dataclasses import dataclass


@dataclass
class Constants:
    product_level_name: str

    stock_name: str

    prediction_name: str

    target_name: str

    date_name: str

    stock_path: str

    target_path: str

    prediction_path: str


constants = Constants(
    product_level_name="product_name",
    stock_name="stocks",
    prediction_name="prediction",
    target_name="counts",
    date_name="date",
    stock_path="modeling/data/aggregated/",
    target_path="modeling/data/aggregated/",
    prediction_path="modeling/data/predictions/",
)
