import os
from collections import deque

from langchain.agents import AgentExecutor, create_gigachat_functions_agent
from langchain_community.chat_models.gigachat import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tg_bot.src.gigachat_tools import new_tools

GIGACHAT_API_KEY = os.environ["GIGACHAT_API_KEY"]

giga = GigaChat(
    credentials=GIGACHAT_API_KEY, verify_ssl_certs=False, model="GigaChat-Pro"
)

agent = create_gigachat_functions_agent(giga, new_tools)
agent_executor = AgentExecutor(agent=agent, tools=new_tools, verbose=True)


global all_chats
all_chats = {}


def respond(message, user_id):

    system_prompt = f"""
            Ты умный бот занимающийся прогнозированием спроса.
            Ты подключен к базе данных в которой хранится информация о товарах.
            Сейчас ты общаешься с пользователем chat_id - {user_id}.
            Ты умеешь делать с помощью функций эти вещи:
            1) Если пользователь хочет узнать назавния товаров используй функцию с именем show_product_names и выведи список товаров.
            2) Для информации по складским остаткам используй функцию с именем stocks_info.
            3) Для прогнозирования используй функцию с именем get_forecast.
            4) Для отправки пользовтелю всех прогнозов в файле используй функцию с именем download_forecast.
            5) Для получения рекомендаций по закупкам используй download_recommendation
            Не придумывай данные сам, бери ответы из функций.
            Расскажи пользователю что ты умеешь по пунктам, но не пиши имена функций.
            Не выдумывай ответы, если у тебя нет данных. 
            Не придумывай данные сам.
            Если каких-то данных не хватает для вызова функции, то нужно спросить данные у пользователя.
            Не отвечай одинаковыми сообщениями.
        """

    memmory = deque(maxlen=3)
    memmory.append(SystemMessage(system_prompt))
    # Проверка, существует ли уже чат для данного пользователя
    if user_id not in all_chats:
        all_chats[user_id] = memmory

    result = agent_executor.invoke(
        {
            "chat_history": list(all_chats[user_id]),
            "input": message,
        }
    )

    all_chats[user_id].append(HumanMessage(content=message))
    all_chats[user_id].append(AIMessage(content=result["output"]))
    all_chats[user_id][0] = SystemMessage(system_prompt)

    return result
