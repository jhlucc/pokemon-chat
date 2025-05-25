import functools
import json
import random
from typing import Optional, Union, Sequence

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from faker import Faker

# ----- 数据库 & ORM -----
Base = declarative_base()


class SalesData(Base):
    __tablename__ = 'sales_data'
    sales_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('product_information.product_id'))
    employee_id = Column(Integer)  # 示例简化
    customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))
    sale_date = Column(String(50))
    quantity = Column(Integer)
    amount = Column(Float)
    discount = Column(Float)


class CustomerInformation(Base):
    __tablename__ = 'customer_information'
    customer_id = Column(Integer, primary_key=True)
    customer_name = Column(String(50))
    contact_info = Column(String(50))
    region = Column(String(50))
    customer_type = Column(String(50))


class ProductInformation(Base):
    __tablename__ = 'product_information'
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(50))
    category = Column(String(50))
    unit_price = Column(Float)
    stock_level = Column(Integer)


class CompetitorAnalysis(Base):
    __tablename__ = 'competitor_analysis'
    competitor_id = Column(Integer, primary_key=True)
    competitor_name = Column(String(50))
    region = Column(String(50))
    market_share = Column(Float)


# ----- 初始化数据库 -----
DATABASE_URI = 'mysql+pymysql://gpt:gpt@localhost:3307/langgraph?charset=utf8mb4'
engine = create_engine(DATABASE_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ----- 工具定义 (增删改查) -----
from pydantic import BaseModel, Field


class AddSaleSchema(BaseModel):
    product_id: int
    employee_id: int
    customer_id: int
    sale_date: str
    quantity: int
    amount: float
    discount: float


class DeleteSaleSchema(BaseModel):
    sales_id: int


class UpdateSaleSchema(BaseModel):
    sales_id: int
    quantity: int
    amount: float


class QuerySalesSchema(BaseModel):
    sales_id: int


@tool(args_schema=AddSaleSchema)
def add_sale(product_id, employee_id, customer_id, sale_date, quantity, amount, discount):
    """Add sale record to the database."""
    session = Session()
    try:
        new_sale = SalesData(
            product_id=product_id,
            employee_id=employee_id,
            customer_id=customer_id,
            sale_date=sale_date,
            quantity=quantity,
            amount=amount,
            discount=discount
        )
        session.add(new_sale)
        session.commit()
        return {"messages": ["销售记录添加成功。"]}
    except Exception as e:
        return {"messages": [f"添加失败，错误原因：{e}"]}
    finally:
        session.close()


@tool(args_schema=DeleteSaleSchema)
def delete_sale(sales_id):
    """Delete sale record from the database."""
    session = Session()
    try:
        sale_to_delete = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_delete:
            session.delete(sale_to_delete)
            session.commit()
            return {"messages": ["销售记录删除成功。"]}
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        return {"messages": [f"删除失败，错误原因：{e}"]}
    finally:
        session.close()


@tool(args_schema=UpdateSaleSchema)
def update_sale(sales_id, quantity, amount):
    """Update sale record in the database."""
    session = Session()
    try:
        sale_to_update = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_update:
            sale_to_update.quantity = quantity
            sale_to_update.amount = amount
            session.commit()
            return {"messages": ["销售记录更新成功。"]}
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        return {"messages": [f"更新失败，错误原因：{e}"]}
    finally:
        session.close()


@tool(args_schema=QuerySalesSchema)
def query_sales(sales_id):
    """Query sale record from the database."""
    session = Session()
    try:
        sale_data = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_data:
            return {
                "sales_id": sale_data.sales_id,
                "product_id": sale_data.product_id,
                "employee_id": sale_data.employee_id,
                "customer_id": sale_data.customer_id,
                "sale_date": sale_data.sale_date,
                "quantity": sale_data.quantity,
                "amount": sale_data.amount,
                "discount": sale_data.discount
            }
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}。"]}
    except Exception as e:
        return {"messages": [f"查询失败，错误原因：{e}"]}
    finally:
        session.close()


repl = PythonREPL()


@tool
def python_repl(code: str):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# ----- 构建两大子代理 (db_agent, code_agent) -----
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants. "
                "Use the provided tools to progress towards answering the question. "
                "If you are unable to fully answer, that's OK, another assistant with different tools "
                "will help where you left off. Execute what you can to make progress. "
                "If you or any of the other assistants have the final answer or deliverable, "
                "prefix your response with 'FINAL ANSWER' so the team knows to stop. "
                "You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt


# 1) 数据库管理员代理
def create_db_agent():
    key = "hk-uomxwi1000053684154a700e0b331d4846fa5bf6fb77ddaf"
    base_url = "https://api.openai-hk.com/v1"
    db_llm = ChatOpenAI(model="gpt-4o", api_key=key, base_url=base_url, temperature=0)
    db_tools = [add_sale, delete_sale, update_sale, query_sales]

    db_prompt = create_agent(db_llm, db_tools, "You should provide accurate data for the code_generator to use.")

    db_agent = db_prompt | db_llm.bind_tools(db_tools)
    return db_agent


# 2) 代码生成/分析代理
def create_code_agent():
    coder_llm = ChatOllama(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:32b",
    )
    code_tools = [python_repl]
    code_prompt = create_agent(coder_llm, code_tools,
                               "Run python code to display diagrams or output execution results.")
    code_agent = code_prompt | coder_llm.bind_tools(code_tools)
    return code_agent


# ----- 构建多代理协作的 StateGraph -----
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal


def agent_node(state, agent, name):
    """将输入消息转给某个子代理，并把该子代理的输出改造成 AIMessage."""
    result = agent.invoke(state)
    if isinstance(result, BaseMessage):
        # 直接是消息
        pass
    else:
        # 强制封装成 AIMessage
        result = AIMessage(content=result.content, additional_kwargs=result.additional_kwargs, name=name)
    return {"messages": [result], "sender": name}


def router(state):
    """路由逻辑：若出现 FINAL ANSWER 或 tool_calls 就转到下一个节点。"""
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content
    if last_message.tool_calls:
        # 有工具调用
        return "call_tool"
    if "FINAL ANSWER" in content or "最终答案" in content:
        return END
    # 轮流在 db_manager <-> code_generator 之间切换
    sender = state.get("sender")
    if sender == "db_manager":
        return "code_generator"
    return "db_manager"


from langgraph.prebuilt import ToolNode


class DataAnalysisAgent:
    def __init__(self):
        self.db_agent = create_db_agent()
        self.code_agent = create_code_agent()
        self.db_node = functools.partial(agent_node, agent=self.db_agent, name="db_manager")
        self.code_node = functools.partial(agent_node, agent=self.code_agent, name="code_generator")
        self.tool_node = ToolNode(tools=[add_sale, delete_sale, update_sale, query_sales, python_repl])

        # 构建 StateGraph
        self.workflow = StateGraph(dict)
        self.workflow.add_node("db_manager", self.db_node)
        self.workflow.add_node("code_generator", self.code_node)
        self.workflow.add_node("call_tool", self.tool_node)

        # 流程：db_manager -> router -> code_generator -> router -> db_manager ...
        self.workflow.add_conditional_edges("db_manager", router, {
            "call_tool": "call_tool", END: END, "db_manager": "db_manager", "code_generator": "code_generator"
        })
        self.workflow.add_conditional_edges("code_generator", router, {
            "call_tool": "call_tool", END: END, "db_manager": "db_manager", "code_generator": "code_generator"
        })

        # 工具调用结束后跳回上一个发消息的节点
        self.workflow.add_conditional_edges("call_tool", lambda s: s["sender"], {
            "db_manager": "db_manager",
            "code_generator": "code_generator"
        })

        self.workflow.set_entry_point("db_manager")
        self.graph = self.workflow.compile(checkpointer=MemorySaver())

    def ask(self, user_input: str):
        """单轮问答：将 user_input 发送给图，直到出现 FINAL ANSWER 或无任务."""
        config = {"configurable": {"thread_id": "session_1"}}
        payload = {
            "messages": [HumanMessage(content=user_input)],
            "sender": "user",
        }
        for chunk in self.graph.stream(payload, config, stream_mode="values"):
            if not self.graph.get_state(config).tasks:
                return chunk["messages"][-1].content

    def chat_loop(self):
        print("欢迎使用数据分析代理，输入'退出'结束")
        while True:
            user_input = input("你：")
            if user_input.lower() == "退出":
                break
            answer = self.ask(user_input)
            print("AI：", answer)


if __name__ == "__main__":
    agent = DataAnalysisAgent()
    agent.chat_loop()
