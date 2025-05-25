from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain

# 1. 连接图谱
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="tczslw278",
    database="neo4j"
)
#  2. 查看图谱结构
print("图谱结构如下:")
print(graph.schema)
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o",
    api_key="hk-uomxwi1000053684154a700e0b331d4846fa5bf6fb77ddaf",
    base_url="https://api.openai-hk.com/v1"
)

# 2. 初始化图谱问答链
cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
)

# 3. 提问
result = cypher_chain.invoke("小火龙哪些地方可以抓到？")

# 4. 打印结果
print("\n【最终回答】")
print(result['result'])
