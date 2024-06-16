import base64
import io
import json
import secrets
import sys
import warnings

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import send_file, Flask
from flask_swagger_ui import get_swaggerui_blueprint

sys.path.append('../')
from modeling.src.training import main as training
from web import html_script

warnings.filterwarnings('ignore')

# URL для доступа к Swagger UI
SWAGGER_URL = '/api/docs'

# URL к файлу спецификации Swagger
API_URL = '/static/swagger.json'


def main():
    server = Flask(__name__)

    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.CYBORG],
        url_base_pathname='/'
    )
    server = app.server

    # Генерация случайного 32-символьного секретного ключа
    secret_key = secrets.token_hex(16)
    app.server.secret_key = secret_key

    # Настройка Swagger UI
    swagger_ui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "Demand Forecasting Dashboard API"
        }
    )

    swagger_link = dbc.Row([
        dbc.Col(
            html.A(
                'API Documentation',
                href='/api/docs',
                target='_blank',
                style={
                    'color': 'white',
                    'textDecoration': 'underline',
                    'cursor': 'pointer'
                }
            ),
            style={'padding': '10px', 'textAlign': 'center'}
        )
    ], style={'margin-top': '10px', 'margin-right': '10px', 'margin-left': '10px', 'margin-bottom': '10px',
            'padding': '0px'})

    server.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

    app.layout = html.Div([

        dbc.Row([
            dbc.Col(
                html.H1('Demand Forecasting', style={'color': 'white'}, ), width=10
            ),
            dbc.Col(
                swagger_link, width=2
            )
        ]),
        html.Br(),
        html.Div(html_script.cards_rigis)
    ], style={'width': '99.5%'})


    @app.callback(
        Output('product-dropdown-chart', 'options'),
        Input('upload-json-data-chart', 'contents'),
        State('upload-json-data-chart', 'filename'),
    )
    def update_product_dropdown_chart(json_data, json_filename):
        """
        Обновляет выпадающий список продуктов для графика на основе загруженных данных JSON.

        Args:
            json_data (str): Содержимое загруженного файла JSON в формате base64.
            json_filename (str): Имя загруженного файла JSON.

        Returns:
            list: Список словарей с метками и значениями для выпадающего списка продуктов,
                отсортированный по убыванию отношения ненулевых значений факта.
        """
        if json_data is None:
            return []

        content_type, content_string = json_data.split(',')
        decoded = base64.b64decode(content_string)
        data = json.loads(decoded)

        product_scores = {}
        for product_id, product_data in data.items():
            fact_values, _ = product_data.get('fact', [[], []])
            if fact_values:
                non_zero_ratio = sum(value != 0 for value in fact_values) / len(fact_values)
                product_scores[product_id] = non_zero_ratio
            else:
                product_scores[product_id] = 0

        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        options = [{'label': product_id, 'value': product_id} for product_id, _ in sorted_products]
        return options


    @app.callback(
        Output('product-dropdown-table', 'options'),
        Input('upload-json-data-table', 'contents'),
        State('upload-json-data-table', 'filename'),
    )
    def update_product_dropdown_table(json_data, json_filename):
        """
        Обновляет выпадающий список продуктов для таблицы на основе загруженных данных JSON.

        Args:
            json_data (str): Содержимое загруженного файла JSON в формате base64.
            json_filename (str): Имя загруженного файла JSON.

        Returns:
            list: Список словарей с метками и значениями для выпадающего списка продуктов,
                отсортированный по убыванию отношения ненулевых значений факта.
        """
        if json_data is None:
            return []

        content_type, content_string = json_data.split(',')
        decoded = base64.b64decode(content_string)
        data = json.loads(decoded)

        product_scores = {}
        for product_id, product_data in data.items():
            fact_values, _ = product_data.get('fact', [[], []])
            if fact_values:
                non_zero_ratio = sum(value != 0 for value in fact_values) / len(fact_values)
                product_scores[product_id] = non_zero_ratio
            else:
                product_scores[product_id] = 0

        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        options = [{'label': product_id, 'value': product_id} for product_id, _ in sorted_products]
        return options


    @app.callback(
        [
            Output('product-chart', 'figure'),
            Output('analysis-text', 'children')
        ],
        [
            Input('product-dropdown-chart', 'value'),
            Input('horizon-dropdown', 'value'),
            State('upload-json-data-chart', 'contents')
        ]
    )
    def update_product_chart(selected_product, selected_horizon, json_data):
        """
        Обновляет график продукта на основе выбранного продукта, горизонта прогнозирования и загруженных данных JSON.

        Args:
            selected_product (str): Выбранный продукт.
            selected_horizon (int): Выбранный горизонт прогнозирования.
            json_data (str): Содержимое загруженного файла JSON в формате base64.

        Returns:
            tuple: Кортеж из двух элементов:
                - plotly.graph_objects.Figure: Объект графика Plotly.
                - str: Текст анализа остатков.
        """
        if json_data is None or selected_product is None:
            return html_script.initial_figure, ""

        content_type, content_string = json_data.split(',')
        decoded = base64.b64decode(content_string)
        data = json.loads(decoded)

        product_data = data.get(selected_product, {})
        if not product_data:
            return html_script.initial_figure, ""

        fig = go.Figure()

        # Добавление fact на график (линия)
        fact_values = product_data.get('fact', [])
        if fact_values:
            fact_values, fact_dates = fact_values
            df_fact = pd.DataFrame({'fact': fact_values}, index=pd.to_datetime(fact_dates))
            fig.add_trace(go.Scatter(
                x=df_fact.index,
                y=df_fact['fact'],
                mode='lines',
                line=dict(color='#FFFFFF', width=2),
                name='Факт',
                yaxis='y'
            ))

        # Добавление predictions на график (линия)
        predictions_values = product_data.get('predictions', [])
        if predictions_values:
            predictions_values, predictions_dates = predictions_values
            if selected_horizon is not None:
                predictions_values = predictions_values[:selected_horizon]
                predictions_dates = predictions_dates[:selected_horizon]
            df_predictions = pd.DataFrame({'predictions': predictions_values}, index=pd.to_datetime(predictions_dates))
            fig.add_trace(go.Scatter(
                x=predictions_dates,
                y=predictions_values,
                mode='lines',
                line=dict(color='#FFFF00', width=2),
                name='Прогноз',
                yaxis='y'
            ))

        # Добавление remains на график (barplot)
        remains_values = product_data.get('remains', [])
        if remains_values:
            remains_values, remains_dates = remains_values
            df_remains = pd.DataFrame({'remains': remains_values}, index=pd.to_datetime(remains_dates))
            fig.add_trace(go.Bar(
                x=df_remains.index,
                y=df_remains['remains'],
                name='Остатки',
                yaxis='y2',
                opacity=0.5,
                marker_color='#808080'
            ))

        # Настройка макета графика
        fig.update_layout(
            plot_bgcolor='#333333',
            paper_bgcolor='#333333',
            font=dict(color='#FFFFFF'),
            title=dict(text=f'Данные продукта {selected_product}', x=0.5),
            xaxis=dict(showgrid=True, gridcolor='#666666', title='Дата'),
            yaxis=dict(showgrid=True, gridcolor='#666666', title='Закупки, шт.'),
            yaxis2=dict(showgrid=True, gridcolor='#666666', title='Остатки, шт.', overlaying='y', side='right'),
            barmode='group'
        )

        # Объединение всех дат
        all_dates = pd.to_datetime(fact_dates).union(pd.to_datetime(predictions_dates)).union(pd.to_datetime(remains_dates))

        # Создание общего DataFrame и заполнение NaN значениями
        product_df = pd.DataFrame(index=all_dates)
        product_df['fact'] = df_fact['fact']
        product_df['predictions'] = df_predictions['predictions']
        product_df['remains'] = df_remains['remains']

        analysis_text = analyse_remains(product_df, selected_product, selected_horizon)

        return fig, analysis_text


    @app.callback(
        Output('product-table', 'data'),
        Input('product-dropdown-table', 'value'),
        State('upload-json-data-table', 'contents'),
    )
    def update_product_table(selected_product, json_data):
        """
        Обновляет таблицу продукта на основе выбранного продукта и загруженных данных JSON.codeArgs:
            selected_product (str): Выбранный продукт.
            json_data (str): Содержимое загруженного файла JSON в формате base64.

        Returns:
            list: Список словарей, представляющих строки таблицы продукта.
        """
        if json_data is None or selected_product is None:
            return []

        content_type, content_string = json_data.split(',')
        decoded = base64.b64decode(content_string)
        data = json.loads(decoded)

        product_data = data.get(selected_product, {})
        if not product_data:
            return []

        fact_values = product_data.get('fact', [])
        predictions_values = product_data.get('predictions', [])
        remains_values = product_data.get('remains', [])

        if fact_values:
            fact_values, fact_dates = fact_values
        else:
            fact_values, fact_dates = [], []

        if predictions_values:
            predictions_values, predictions_dates = predictions_values
        else:
            predictions_values, predictions_dates = [], []

        if remains_values:
            remains_values, remains_dates = remains_values
        else:
            remains_values, remains_dates = [], []

        # Получение последних 12 месяцев факта и остатка
        last_12_months_fact_dates = fact_dates[-12:]
        last_12_months_remains_dates = remains_dates[-12:]

        # Объединение последних 12 месяцев факта, остатка и всех дат прогноза
        all_dates = sorted(set(last_12_months_fact_dates + last_12_months_remains_dates + predictions_dates))

        table_data = []
        for date in all_dates:
            row = {
                'Дата': date,
                'Predictions': predictions_values[predictions_dates.index(date)] if date in predictions_dates else None,
                'Fact': fact_values[fact_dates.index(date)] if date in fact_dates else None,
                'Remains': remains_values[remains_dates.index(date)] if date in remains_dates else None
            }
            table_data.append(row)

        return table_data


    @app.callback(
        Output('train-button', 'n_clicks'),
        [Input('train-button', 'n_clicks')]
    )
    def train_button_callback(n_clicks):
        """
        Обработчик нажатия кнопки "Обучить".
        codeArgs:
            n_clicks (int): Количество нажатий на кнопку.

        Returns:
            int: Количество нажатий на кнопку.
        """
        if n_clicks:
            print("Train button clicked")
            training()
        return n_clicks


    @server.route('/download_recommendations/<filename>')
    def download_recommendations(filename):
        """
        Обработчик загрузки файла рекомендаций JSON.
        Copy codeArgs:
            filename (str): Имя загружаемого файла рекомендаций.

        Returns:
            flask.Response: Объект ответа Flask с загружаемым файлом рекомендаций JSON.
        """
        json_data = flask.session.get('json_data_chart', None)
        print(f"JSON data retrieved from session: {json_data}")  # Добавьте эту строку

        if json_data is None:
            return ''

        recommendations_data = generate_recommendations_json(json_data)
        json_string = json.dumps(recommendations_data, ensure_ascii=False)
        mem_file = io.BytesIO(json_string.encode('utf-8'))

        return send_file(
            mem_file,
            as_attachment=True,
            download_name=filename,
            mimetype='application/json'
        )


    @app.callback(
        Output('download-recommendations', 'data'),
        Input('download-recommendations-button', 'n_clicks'),
        State('upload-json-data-chart', 'contents'),
        State('upload-json-data-chart', 'filename'),
        prevent_initial_call=True
    )
    def download_recommendations_callback(n_clicks, json_data, json_filename):
        """
        Обработчик загрузки файла рекомендаций JSON.
        Copy codeArgs:
            n_clicks (int): Количество нажатий на кнопку загрузки рекомендаций.
            json_data (str): Содержимое загруженного файла JSON в формате base64.
            json_filename (str): Имя загруженного файла JSON.

        Returns:
            dict: Словарь с содержимым файла рекомендаций JSON и именем файла.
        """
        if n_clicks is None:
            raise PreventUpdate
        content_type, content_string = json_data.split(',')
        decoded = base64.b64decode(content_string)
        data = json.loads(decoded)
        recommendations_data = generate_recommendations_json(data)
        json_string = json.dumps(recommendations_data, ensure_ascii=False)
        return dict(content=json_string, filename=f"recommendations_{json_filename}")


    def analyse_remains(product_df, product_name, horizont):
        """
        Анализирует остатки продукта и генерирует текстовое описание.
        Copy codeArgs:
            product_df (pd.DataFrame): DataFrame с данными продукта.
            product_name (str): Название продукта.
            horizont (int): Горизонт прогнозирования.

        Returns:
            str: Текстовое описание анализа остатков.
        """
        STOCKS_NAME = 'remains'
        PREDICTION_NAME = 'predictions'
        product_df = product_df.reset_index()
        product_df = product_df.rename(columns={'index': 'date'})
        product_df = product_df.sort_values(by=['date'], ascending=False)

        fact_df = product_df[product_df['fact'].notna()]
        predictions_df = product_df[product_df['predictions'].notna()]
        remains_df = product_df[product_df['remains'].notna()]

        stocks_analyze = round((remains_df[STOCKS_NAME] - predictions_df[PREDICTION_NAME].sum()).values[0])

        if horizont == None:
            horizont = 12

        if stocks_analyze < 0:
            show_text = f"""
            Вам нужно закупить {abs(stocks_analyze)} штук и 10% - {round(abs(stocks_analyze) * 0.1)} страховой запас.
            Сейчас на складах осталось {remains_df[STOCKS_NAME].values[0]} штук товара {product_name}.
            В ближайшие {horizont} периодов купят {round(predictions_df[PREDICTION_NAME].sum())} штук товара.
            """

        else:
            show_text = f"""
            У вас достаточно товара для удовлетоврения спроса.
            Сейчас на складах осталось {remains_df[STOCKS_NAME].values[0]} штук товара {product_name}.
            В ближайшие {horizont} периодов купят {round(predictions_df[PREDICTION_NAME].sum())} штук товара.
            """
        return show_text


    def analyse_remains_json(product_df, product_name, horizont):
        """
        Анализирует остатки продукта и возвращает результаты в формате JSON.
        Copy codeArgs:
            product_df (pd.DataFrame): DataFrame с данными продукта.
            product_name (str): Название продукта.
            horizont (int): Горизонт прогнозирования.

        Returns:
            dict: Словарь с результатами анализа остатков.
        """
        STOCKS_NAME = 'remains'
        PREDICTION_NAME = 'predictions'
        product_df = product_df.reset_index()
        product_df = product_df.sort_values(by=['date'], ascending=False)

        predictions_df = product_df[product_df['predictions'].notna()]
        remains_df = product_df[product_df['remains'].notna()]

        stocks_analyze = round((remains_df[STOCKS_NAME] - predictions_df[PREDICTION_NAME].sum()).values[0])
        stocks_current = remains_df[STOCKS_NAME].values[0]
        predictions_sum = round(predictions_df[PREDICTION_NAME].sum())

        return {
            'horizont': horizont,
            'stocks_analyze': stocks_analyze,
            'stocks_current': stocks_current,
            'predictions_sum': predictions_sum
        }


    def generate_recommendations_json(json_data):
        """
        Генерирует JSON-файл с рекомендациями на основе загруженных данных JSON.
        Copy codeArgs:
            json_data (dict): Словарь с загруженными данными JSON.

        Returns:
            dict: Словарь с рекомендациями для каждого продукта.
        """
        if json_data is None:
            return {}

        recommendations_data = {}

        for product_name in json_data.keys():
            product_data = json_data[product_name]
            product_df = pd.DataFrame(columns=['date', 'fact', 'predictions', 'remains'])

            fact_values, fact_dates = product_data.get('fact', [[], []])
            predictions_values, predictions_dates = product_data.get('predictions', [[], []])
            remains_values, remains_dates = product_data.get('remains', [[], []])

            product_df['date'] = fact_dates + predictions_dates + remains_dates
            product_df['fact'] = fact_values + [None] * (len(product_df) - len(fact_values))
            product_df['predictions'] = [None] * len(fact_values) + predictions_values + [None] * (
                    len(product_df) - len(fact_values) - len(predictions_values))
            product_df['remains'] = [None] * (len(product_df) - len(remains_values)) + remains_values

            analysis_data = analyse_remains_json(product_df, product_name, len(predictions_values))

            recommendations_data[product_name] = analysis_data

        return recommendations_data


    @app.callback(
        Output('upload-json-data-chart-text', 'children'),
        [Input('upload-json-data-chart', 'filename')]
    )
    def update_chart_upload_text(filename):
        """
        Обновляет текст загрузки файла для графика.
        Args:
            filename (str): Имя загруженного файла.

        Returns:
            str: Текст с информацией о загруженном файле или приглашение к загрузке файла.
        """
        if filename:
            return f'Загруженный файл: {filename}'
        return 'Перетащите JSON файл или выберите файл'


    @app.callback(
        Output('download-json-data', 'data'),
        Input('download-json-button', 'n_clicks'),
        State('product-table', 'data'),
        State('product-dropdown-table', 'value'),
        prevent_initial_call=True
    )
    def download_json_data(n_clicks, table_data, selected_product):
        """
        Callback для загрузки данных в формате JSON.
        Args:
            n_clicks (int): Количество нажатий на кнопку загрузки данных.
            table_data (list): Данные таблицы продукта.
            selected_product (str): Выбранный продукт.

        Returns:
            dict: Словарь с содержимым файла JSON и именем файла.
        """
        if n_clicks is None:
            raise PreventUpdate
        modified_data = {
            selected_product: {
                'fact': [[], []],
                'predictions': [[], []],
                'remains': [[], []]
            }
        }

        for row in table_data:
            date = row['Дата']
            fact = float(row['Fact']) if row['Fact'] else None
            prediction = float(row['Predictions']) if row['Predictions'] else None
            remain = float(row['Remains']) if row['Remains'] else None

            modified_data[selected_product]['fact'][0].append(fact)
            modified_data[selected_product]['fact'][1].append(date)
            modified_data[selected_product]['predictions'][0].append(prediction)
            modified_data[selected_product]['predictions'][1].append(date)
            modified_data[selected_product]['remains'][0].append(remain)
            modified_data[selected_product]['remains'][1].append(date)

        json_data = json.dumps(modified_data, ensure_ascii=False)

        return dict(content=json_data, filename=f"data_{selected_product}.json")

    app.run_server()


# Запуск приложения
if __name__ == "__main__":
    main()
