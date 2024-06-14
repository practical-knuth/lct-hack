import warnings

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

warnings.filterwarnings("ignore")

color_cards = "#1f1f1f"
color_text = "#ffffff"
plot_bgcolor = "#ffffff"
paper_bgcolor = "#1f1f1f"
table_header_color = "rgb(30, 30, 30)"
table_back_color = "rgb(50, 50, 50)"
table_grid_color = "1px solid #494949"

style_cell = {
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "maxWidth": "100%",
    "textAlign": "left",
}
style_table = {
    "maxHeight": "100%",
    "maxWidth": "98%",
}
style_header = {
    "backgroundColor": table_header_color,
    "color": "white",
    "border": table_grid_color,
}
style_data = {
    "backgroundColor": table_back_color,
    "color": "white",
    "border": table_grid_color,
    "whiteSpace": "normal",
    "height": "auto",
}

text_size = 21

product_dropdown = dcc.Dropdown(
    id="product-dropdown",
    options=[],
    value=None,
    placeholder="Выберите продукт",
    style={"width": "500px", "margin": "10px"},
)

initial_figure = {
    "data": [],
    "layout": {
        "plot_bgcolor": "#333333",
        "paper_bgcolor": "#333333",
        "font": {"color": "#FFFFFF"},
        "xaxis": {"showgrid": True, "gridcolor": "#666666", "title": "Дата"},
        "yaxis": {"showgrid": True, "gridcolor": "#666666", "title": "Продажи, шт."},
        "yaxis2": {
            "showgrid": True,
            "gridcolor": "#666666",
            "title": "Остатки, шт.",
            "overlaying": "y",
            "side": "right",
        },
        "barmode": "group",
        "title": {"text": "Данные продукта", "x": 0.5},
    },
}
product_chart = dcc.Graph(
    id="product-chart", figure=initial_figure, style={"height": "500px"}
)

product_table = dash_table.DataTable(
    id="product-table",
    columns=[
        {"name": "Дата", "id": "Дата", "editable": False},
        {"name": "Predictions", "id": "Predictions", "editable": True},
        {"name": "Fact", "id": "Fact", "editable": True},
        {"name": "Remains", "id": "Remains", "editable": True},
    ],
    data=[],
    style_cell={
        "textAlign": "center",
        "backgroundColor": table_back_color,
        "color": color_text,
    },
    style_header={
        "backgroundColor": table_header_color,
        "fontWeight": "bold",
        "color": color_text,
    },
    style_data_conditional=[
        {
            "if": {"column_id": "Predictions"},
            "backgroundColor": "rgba(175, 250, 178, 0.4)",  # Прозрачно-красный цвет
        }
    ],
    editable=True,
)

load_json_data_chart = dcc.Upload(
    id="upload-json-data-chart",
    children=html.Div(
        [
            "Перетащите JSON файл или ",
            html.A(
                "выберите файл",
                style={
                    "color": "blue",
                    "text-decoration": "underline",
                    "cursor": "pointer",
                },
            ),
            html.Div(id="upload-json-data-chart-text", style={"margin-top": "10px"}),
        ]
    ),
    style={
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
    },
    multiple=False,
)

load_json_data_table = dcc.Upload(
    id="upload-json-data-table",
    children=html.Div(
        [
            "Перетащите JSON файл или ",
            html.A(
                "выберите файл",
                style={
                    "color": "blue",
                    "text-decoration": "underline",
                    "cursor": "pointer",
                },
            ),
            html.Div(id="upload-json-data-table-text", style={"margin-top": "10px"}),
        ]
    ),
    style={
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
    },
    multiple=False,
)

product_dropdown_chart = dcc.Dropdown(
    id="product-dropdown-chart",
    options=[],
    value=None,
    placeholder="Выберите продукт",
    style={"width": "500px", "margin": "10px"},
)

product_dropdown_table = dcc.Dropdown(
    id="product-dropdown-table",
    options=[],
    value=None,
    placeholder="Выберите продукт",
    style={"width": "500px", "margin": "10px"},
)

train_button_style = {
    "backgroundColor": "rgba(252, 91, 107, 0.7)",  # Нежно пастельно-красный цвет с прозрачностью
    "color": "white",
    "borderRadius": "10px",
    "padding": "10px 20px",
    "margin": "10px",
    "border": "none",
    "fontSize": "18px",
    "fontWeight": "bold",
    "cursor": "pointer",
    "width": "400px",
}

download_button = html.A(
    html.Button(
        "Скачать JSON",
        style={
            "backgroundColor": "rgba(175, 250, 178, 0.6)",
            "color": "white",
            "borderRadius": "20px",
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
            "transition": "all 0.3s ease",
            "marginBottom": "10px",
            "border": "none",
            "padding": "10px 20px",
            "fontWeight": "bold",
            "cursor": "pointer",
        },
    ),
    id="download-json-link",
    href="",
    target="_blank",
)

download_recommendations_button = html.Div(
    [
        html.Button(
            "Скачать рекомендации (JSON)",
            id="download-recommendations-button",
            style={
                "backgroundColor": "rgba(175, 250, 178, 0.6)",
                "color": "white",
                "borderRadius": "20px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                "transition": "all 0.3s ease",
                "marginBottom": "10px",
                "border": "none",
                "padding": "10px 20px",
                "fontWeight": "bold",
                "cursor": "pointer",
            },
        ),
        dcc.Download(id="download-recommendations"),
    ]
)

horizon_dropdown = dcc.Dropdown(
    id="horizon-dropdown",
    options=[{"label": str(i), "value": i} for i in range(1, 13)],
    value=None,
    placeholder="Выберите горизонт",
    style={"width": "200px", "margin": "10px"},
)

cards_rigis = [
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.B(
                                    "Визуализация данных продукта",
                                    style={"font-size": 18, "color": color_text},
                                ),
                                html.Div(
                                    load_json_data_chart, style={"margin": "10px"}
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            product_dropdown_chart,
                                            width=3,
                                            style={
                                                "margin-right": "5px",
                                                "margin-left": "10px",
                                            },
                                        ),
                                        dbc.Col(
                                            horizon_dropdown,
                                            width=2,
                                            style={"margin-left": "5px"},
                                        ),
                                    ],
                                    align="start",
                                ),
                                html.Div(product_chart, style={"margin": "10px"}),
                                html.Div(
                                    id="analysis-text",
                                    style={
                                        "color": "white",
                                        "background-color": "rgba(175, 250, 178, 0.6)",
                                        "border-radius": "15px",
                                        "padding": "15px",
                                        "margin-top": "10px",
                                        "font-size": "20px",  # Размер шрифта 20
                                        "font-family": "Arial, sans-serif",  # Шрифт Arial
                                    },
                                ),  # Новый элемент для текстового сообщения с стилями
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [download_recommendations_button],
                                            style={
                                                "padding": "10px",
                                                "textAlign": "center",
                                            },
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    color=color_cards,
                    style={"height": "auto", "border-radius": "0", "padding": "10px"},
                ),
                style={
                    "padding-right": "10px",
                    "padding-left": "10px",
                    "padding-bottom": "0px",
                },
            ),
        ],
        style={
            "margin-top": "10px",
            "margin-right": "10px",
            "margin-left": "10px",
            "margin-bottom": "10px",
            "padding": "0px",
        },
    ),
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.B(
                                    "Данные продукта",
                                    style={"font-size": 18, "color": color_text},
                                ),
                                html.Div(
                                    load_json_data_table, style={"margin": "10px"}
                                ),
                                html.Div(
                                    product_dropdown_table, style={"margin": "10px"}
                                ),
                                html.Div(product_table, style={"margin": "10px"}),
                            ]
                        ),
                    ],
                    color=color_cards,
                    style={"height": "auto", "border-radius": "0", "padding": "10px"},
                ),
                style={
                    "padding-right": "10px",
                    "padding-left": "10px",
                    "padding-bottom": "0px",
                },
            ),
        ],
        style={
            "margin-top": "10px",
            "margin-right": "10px",
            "margin-left": "10px",
            "margin-bottom": "10px",
            "padding": "0px",
        },
    ),
]

train_button = html.Button("Train", id="train-button", style=train_button_style)

cards_rigis.append(
    dbc.Row(
        [dbc.Col(download_button, style={"padding": "10px", "textAlign": "center"})]
    )
)

train_button_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.B("Обучение модели", style={"font-size": 18, "color": color_text}),
                html.P(
                    "Данный раздел отвечает за переобучение моделей. При необходимости можно воспользоваться данной функцией при помощи клика.",
                    style={
                        "color": color_text,
                        "font-size": "16px",
                        "margin-bottom": "10px",
                        "border": "2px dashed #666666",  # Пунктирная граница
                        "border-radius": "10px",  # Закругленные края
                        "padding": "10px",  # Внутренний отступ
                        "background-color": "#1f1f1f",  # Цвет фона
                        "font-family": "Arial, sans-serif",  # Шрифт Arial
                        "textAlign": "center",
                    },
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                id="loading",
                                type="circle",
                                color="rgba(175, 250, 178, 0.6)",  # Цвет индикатора загрузки
                                style={"margin": "10px"},
                                children=[train_button],
                            ),
                            style={"padding": "10px", "textAlign": "center"},
                        )
                    ]
                ),
            ]
        )
    ],
    color=color_cards,
    style={"height": "auto", "border-radius": "0", "padding": "10px"},
)

train_button_row = dbc.Row(
    [
        dbc.Col(
            train_button_card,
            style={
                "padding-right": "10px",
                "padding-left": "10px",
                "padding-bottom": "0px",
            },
        )
    ],
    style={
        "margin-top": "10px",
        "margin-right": "10px",
        "margin-left": "10px",
        "margin-bottom": "10px",
        "padding": "0px",
    },
)

cards_rigis.append(train_button_row)
