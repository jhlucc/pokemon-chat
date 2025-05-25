import os
import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# ---------- 数据库初始化 ----------
Base = declarative_base()

DATABASE_URI = 'mysql+pymysql://gpt:gpt@localhost:3307/langgraph?charset=utf8mb4'
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)


class FileImportSchema(BaseModel):
    """
    用于导入文件的参数：
    - file_path: 文件路径 (CSV 或 JSON)
    - table_name: 要在数据库中创建或替换的表名
    """
    file_path: str = Field(description="文件路径，可以是 CSV 或 JSON")
    table_name: str = Field(description="数据库中的表名")


@tool(args_schema=FileImportSchema)
def import_data_to_db(file_path: str, table_name: str):
    """
    读取 file_path（CSV 或 JSON 格式）中的数据，根据数据结构
    在数据库中创建对应的表，然后将数据插入到 table_name 表中。
    如果表已存在，则替换或追加。
    """
    session = Session()
    try:
        # 判定文件是否存在
        if not os.path.exists(file_path):
            return {"messages": [f"文件 {file_path} 不存在！"]}

        # 自动根据后缀判断读取CSV还是JSON
        file_ext = os.path.splitext(file_path)[-1].lower()

        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
        else:
            return {"messages": [f"不支持的文件类型: {file_ext}，仅支持 CSV 或 JSON"]}

        # 利用 pandas 的 to_sql 方法自动创建表结构并插入数据
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)

        return {"messages": [f"文件 {file_path} 中的数据已成功写入 {table_name} 表。"]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"数据写入失败：{e}"]}
    finally:
        session.close()


# ---------- Agent 定义 ----------
class DataAgent:
    def __init__(self):
        # 将所有工具放进 ToolNode
        self.tools = [import_data_to_db]
        self.tool_node = ToolNode(self.tools)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key="hk-uomxwi1000053684154a700e0b331d4846fa5bf6fb77ddaf",
            base_url="https://api.openai-hk.com/v1",
            temperature=0
        ).bind_tools(self.tools)

        # 构建对话状态机
        self.graph = self._build_workflow()

    def _call_model(self, state):
        """
        让 LLM 处理当前消息并给出回复。
        """
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state):
        """
        判断下一步走向：
        - 如果没有 tool_call，说明没有工具要调用，直接结束。
        - 如果有 tool_call，就进入 run_tool。
        """
        last_msg = state["messages"][-1]
        if not last_msg.tool_calls:
            return "end"
        return "run_tool"

    def _run_tool(self, state):
        """
        执行工具并返回执行结果。
        """
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
        """
        构建一个简单的状态机：
        START -> agent -> 根据 tool_calls 判断 -> run_tool -> agent -> ...
        如果没有 tool_calls 则 END。
        """
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", self.tool_node)
        workflow.add_node("run_tool", self._run_tool)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, {
            "run_tool": "run_tool",
            "end": END
        })
        workflow.add_edge("run_tool", "agent")
        workflow.add_edge("action", "agent")

        # 注意，这里将 ToolNode 放置在 agent 之后，
        # 让 LLM 有机会先决定是否需要 Tool，再进行调用
        return workflow.compile(checkpointer=MemorySaver())

    def ask(self, question: str) -> str:
        """
        单次问答交互，返回最终答案。
        """
        config = {"configurable": {"thread_id": "session_1"}}
        user_message = {"role": "user", "content": question}
        for chunk in self.graph.stream({"messages": [user_message]}, config, stream_mode="values"):
            state = self.graph.get_state(config)
            if not state.tasks:
                return chunk["messages"][-1].content

    def chat_loop(self):
        """
        循环对话模式，用户可以多轮提问。
        """
        config = {"configurable": {"thread_id": "session_1"}}
        print("欢迎使用数据导入 Agent，输入 '退出' 结束对话")
        while True:
            user_input = input("你：")
            if user_input.lower() in ["退出", "quit", "exit"]:
                print("再见！")
                break
            user_message = {"role": "user", "content": user_input}
            for chunk in self.graph.stream({"messages": [user_message]}, config, stream_mode="values"):
                state = self.graph.get_state(config)
                if not state.tasks:
                    # 输出 AI 最后的回复
                    print("AI：", chunk["messages"][-1].content)
                    break


if __name__ == "__main__":
    agent = DataAgent()
    agent.chat_loop()
