from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_core.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
import requests
import json
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# ---------- 数据库定义 ----------
Base = declarative_base()
CITY_ALIAS = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "深圳": "Shenzhen",
    "广州": "Guangzhou",
    "杭州": "Hangzhou",
    "南京": "Nanjing",
    "苏州": "Suzhou",
    "常州": "Changzhou",
    "无锡": "Wuxi",
    "宁波": "Ningbo"
}


def translate_city_name(name: str) -> str:
    return CITY_ALIAS.get(name.strip(), name.strip())


class Weather(Base):
    __tablename__ = 'weather'
    city_id = Column(Integer, primary_key=True)
    city_name = Column(String(50))
    main_weather = Column(String(50))
    description = Column(String(100))
    temperature = Column(Float)
    feels_like = Column(Float)
    temp_min = Column(Float)
    temp_max = Column(Float)


DATABASE_URI = 'mysql+pymysql://gpt:gpt@localhost:3307/langgraph?charset=utf8mb4'
engine = create_engine(DATABASE_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


# ---------- Pydantic Schemas ----------
class WeatherLoc(BaseModel):
    location: str = Field(description="城市名称")


class WeatherInfo(BaseModel):
    city_id: int
    city_name: str
    main_weather: str
    description: str
    temperature: float
    feels_like: float
    temp_min: float
    temp_max: float


class QueryWeatherSchema(BaseModel):
    city_name: str


class DeleteWeatherSchema(BaseModel):
    city_name: str


@tool(args_schema=WeatherLoc)
def get_weather(location):
    """根据城市名获取实时天气信息"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    location = translate_city_name(location)
    params = {
        "q": location,
        "appid": "ee5204216d6c4f500610967c11211409",
        "units": "metric",
        "lang": "zh_cn"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return json.dumps(data)


@tool(args_schema=WeatherInfo)
def insert_weather_to_db(city_id, city_name, main_weather, description, temperature, feels_like, temp_min, temp_max):
    """将天气信息插入数据库（如果存在则更新）"""
    session = Session()
    try:
        existing = session.query(Weather).filter(Weather.city_id == city_id).first()
        if existing:
            existing.city_name = city_name
            existing.main_weather = main_weather
            existing.description = description
            existing.temperature = temperature
            existing.feels_like = feels_like
            existing.temp_min = temp_min
            existing.temp_max = temp_max
            msg = "天气数据已更新至数据库。"
        else:
            weather = Weather(
                city_id=city_id,
                city_name=city_name,
                main_weather=main_weather,
                description=description,
                temperature=temperature,
                feels_like=feels_like,
                temp_min=temp_min,
                temp_max=temp_max
            )
            session.add(weather)
            msg = "天气数据已成功插入数据库。"
        session.commit()
        return {"messages": [msg]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"数据写入失败：{e}"]}
    finally:
        session.close()


@tool(args_schema=QueryWeatherSchema)
def query_weather_from_db(city_name: str):
    """根据城市名查询数据库中的天气信息"""
    session = Session()
    try:
        data = session.query(Weather).filter(Weather.city_name == city_name).first()
        if data:
            return data.__dict__
        else:
            return {"messages": [f"未找到城市 {city_name} 的数据"]}
    except Exception as e:
        return {"messages": [f"查询失败：{e}"]}
    finally:
        session.close()


@tool(args_schema=DeleteWeatherSchema)
def delete_weather_from_db(city_name: str):
    """根据城市名删除数据库中的天气信息"""
    session = Session()
    try:
        data = session.query(Weather).filter(Weather.city_name == city_name).first()
        if data:
            session.delete(data)
            session.commit()
            return {"messages": [f"已删除城市 {city_name} 的天气信息"]}
        else:
            return {"messages": [f"未找到城市 {city_name} 的数据"]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"删除失败：{e}"]}
    finally:
        session.close()


class WeatherAgent:
    def __init__(self):
        self.tools = [get_weather, insert_weather_to_db, query_weather_from_db, delete_weather_from_db]
        self.tool_node = ToolNode(self.tools)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key="hk-uomxwi1000053684154a700e0b331d4846fa5bf6fb77ddaf",
            base_url="https://api.openai-hk.com/v1",
            temperature=0
        ).bind_tools(self.tools)
        self.graph = self._build_workflow()

    def _call_model(self, state):
        messages = state["messages"]
        # print("[调试] LLM 接收到的消息:", messages)
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state):
        last_msg = state["messages"][-1]
        if not last_msg.tool_calls:
            return "end"
        elif last_msg.tool_calls[0]["name"] == "delete_weather_from_db":
            return "run_tool"
        else:
            return "continue"

    def _run_tool(self, state):
        new_messages = []
        tool_calls = state["messages"][-1].tool_calls
        tool_map = {t.name: t for t in self.tools}

        for call in tool_calls:
            tool = tool_map.get(call["name"])
            if tool:
                result = tool.invoke(call["args"])
                new_messages.append({
                    "role": "tool",
                    "name": call["name"],
                    "content": result,
                    "tool_call_id": call["id"]
                })
        return {"messages": new_messages}

    def _build_workflow(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", self.tool_node)
        workflow.add_node("run_tool", self._run_tool)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, {
            "continue": "action",
            "run_tool": "run_tool",
            "end": END,
        })
        workflow.add_edge("action", "agent")
        workflow.add_edge("run_tool", "agent")
        return workflow.compile(checkpointer=MemorySaver(), interrupt_before=["run_tool"])

    def ask(self, question: str) -> str:
        config = {"configurable": {"thread_id": "session_1"}}
        user_message = {"role": "user", "content": question}
        for chunk in self.graph.stream({"messages": [user_message]}, config, stream_mode="values"):
            state = self.graph.get_state(config)
            if not state.tasks:
                return chunk["messages"][-1].content

    def chat_loop(self):
        config = {"configurable": {"thread_id": "session_1"}}
        print("欢迎使用天气对话助手，输入'退出'结束对话")
        while True:
            user_input = input("你：")
            if user_input.lower() == "退出":
                break
            user_message = {"role": "user", "content": user_input}
            for chunk in self.graph.stream({"messages": [user_message]}, config, stream_mode="values"):
                state = self.graph.get_state(config)
                if not state.tasks:
                    # print("AI：", chunk["messages"][-1].content)
                    break
                if state.tasks[0].name == "run_tool":
                    confirm = input("是否允许执行删除操作？(是/否): ")
                    if confirm == "是":
                        self.graph.update_state(config=config, values=chunk)
                        for event in self.graph.stream(None, config, stream_mode="values"):
                            print("AI：", event["messages"][-1].content)
                    else:
                        tool_call_id = state.values["messages"][-1].tool_calls[0]["id"]
                        new_message = {
                            "role": "tool",
                            "name": "delete_weather_from_db",
                            "content": "管理员不允许删除操作！",
                            "tool_call_id": tool_call_id
                        }
                        self.graph.update_state(config, {"messages": [new_message]}, as_node="run_tool")
                        for event in self.graph.stream(None, config, stream_mode="values"):
                            print("AI：", event["messages"][-1].content)


if __name__ == "__main__":
    agent = WeatherAgent()
    agent.chat_loop()
