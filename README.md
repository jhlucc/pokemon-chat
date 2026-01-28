                          
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>  
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=it">Italiano</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=jhlucc&project=pokemon-chat&lang=id">Bahasa Indonesia</a>
      </div>
    </div>
   </details> 

</div> 

📘 中文 | [English](./README.en.md)

<img src="resources/picture/11.png" alt="图标" style="width: 27%;" />

# 「可萌」 <img src="resources/picture/brain-removebg-preview.png" alt="图标" style="width: 11%;" /> 基于知识库与知识图谱的专域聊天助手

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=ffffff)
![LangGraph](https://img.shields.io/badge/LangGraph-Flow-green?style=flat&logo=databricks)
![GraphRAG](https://img.shields.io/badge/GraphRAG-KG-blueviolet?style=flat)
![Agent](https://img.shields.io/badge/Agent-System-orange?style=flat)
![Vue3](https://img.shields.io/badge/Vue-3.0-4FC08D?style=flat&logo=vue.js)
![License](https://img.shields.io/github/license/bitcookies/winrar-keygen.svg?logo=github)

 

<img src="resources/picture/img.png" alt="图标" style="width: 90%;" />



---

## 📝 项目介绍

宝可梦（Pokémon）作为全球最具影响力的 IP 之一，拥有庞大的世界观设定与海量角色数据。在游戏、动画、卡牌、电影等多领域的多年积累下，其知识体系庞杂且高度结构化，非常适合应用于知识图谱建模与智能问答场景。

随着大语言模型（LLM）与知识增强技术的发展，将宝可梦宇宙构建为一个**多模态、结构化、可交互的 AI 系统**成为可能。本项目以 百度贴吧 与维基百科等数据源为基础，构建出覆盖宝可梦角色、属性、技能、地区、演化路径等元素的知识图谱，并结合大模型能力，打造一个**专属宝可梦世界的智能对话助手** ——「可萌」。

在此基础上，我们融合了 **LangGraph 推理流程编排**、 **GraphRAG 检索增强技术**，以及**知识图谱可视化探索能力**，使用户不仅可以通过自然语言提问获得精确答案，还能以图谱形式直观探索宝可梦世界。同时支持基于地理位置的地图定位功能，将宝可梦世界与真实世界坐标一一映射，实现 **宝可梦地点知识的空间可视化** :earth_asia: 。

​        本项目致力于打造一个可迁移、可扩展、面向爱好者的**专域智能助手模板系统**，你可以轻松将其迁移至其他角色中打造专域的智能助手，仅需更换知识源与图谱结构，即可实现高质量的语义问答与可视化知识探索体验。

---

## 🚀 新增特性 (New Features)

- **LangChain & LangGraph**: 重构为LangChain 1.x 与 LangGraph 1.0 多智能体编排对话助手
- **LightRAG Integration**: 支持 LightRAG 高效检索图谱构建
- **Advanced RAG**: 集成 Self-RAG、CRAG、HyDE 与 Query Decomposition 高级rag
- **Agentic Memory**: 具备用户偏好自适应的长时记忆能力与时间规划
- **MCP Service**: 支持 Model Context Protocol 连接真实世界地理数据
- **Performance**: 内置 Semantic Cache 与 Speculative RAG 加速生成

---

## 🎯系统架构

通过本项目的实施，我们不仅完成了vue3+fastapi的一个完整项目，同时构建了一个基于宝可梦知识图谱的智能问答系统。积累了语义结构建模如bert+tf-idf+规则匹配机制、以及图谱融合与生成式问答的丰富实践经验。系统支持对宝可梦的进化关系、属性克制、技能特征、地理分布等内容进行精准问答，极大提升了用户在交互式探索中的体验感。

架构核心：
- **混合检索**: 向量检索 (Milvus) + 图谱检索 (Neo4j) + 关键词检索 (BM25)
- **智能体编排**: LangGraph1.x 状态机管理复杂任务流
- **知识增强**: GraphRAG 提取实体关系

以下是本项目的核心技术架构图：

 <img src="resources/picture/now.png" alt="图标" style="width: 100%;" />

##  🎯项目特色

1. 基于爬取的数据微调了基于宝可梦的专域大模型——[可萌](https://huggingface.co/qwqqwq/qwen2.5-14b-instruct-pokemon-int4) 。
2. 基于爬取数据构建了宝可梦知识图谱（维基百科）。
3. 自动化标注训练NER数据，使用roberta+TF-IDF+规则匹配来命中图谱中的实体与属性。
4. 使用FunASR来实现ASR功能（阿里达摩院开源语音识别）。
5. 实现 MCP 服务，支持宝可梦世界与真实世界地理坐标的映射查询。
6. 抽取RAGflow中的deepdoc来强化知识库的解析和抽取能力。
7. 使用LangGraph实现多智能体协同（RAG + Search + Graph + MCP）。
8. 封装agent 基类实现多智能体功能。
9. 支持知识图谱搜索、网络搜索、知识库搜索、MCP搜索、语音搜索，可以同时集成也可以任选其一。

---

## 🛠️ 部署指南 (Deployment)

> **前置要求**：已安装 Docker / Docker Compose

### 🐳 Docker Compose 一键启动（推荐）

无需手动配置复杂环境，直接使用 Docker Compose 启动所有服务：

```bash
# 1. 克隆仓库
git clone https://github.com/skygazer42/pokemon-chat.git
cd pokemon-chat

# 2. 配置环境变量（后端统一从仓库根目录 `.env` 读取，Docker Compose 也会加载该文件）
cp .env.example .env
# 编辑 .env，填写 LLM API KEY（如 llm_api_key / SILICONFLOW_API_KEY）
# 可选：开启检索/工具能力（示例）
#   enable_knowledge_graph=true   # Neo4j 知识图谱
#   enable_knowledge_base=true    # Milvus 知识库
#   enable_web_search=true        # Web 搜索（需 tavily_api_key）
#   enable_mcp=true               # MCP（通常配合 `--profile infra --profile mcp`）
# 可选：开启语音识别（FunASR）-> enable_asr=true，funasr_url=ws://funasr:10095（Docker）
# 可选：限制 CORS 来源（生产环境建议）-> cors_allow_origins=http://localhost:3100

# 3. 启动服务
cd docker
# 默认仅启动 App（API + Web）。
docker compose up -d --build

# 可选：启动 Infra（Neo4j/MySQL/Milvus 等）+ 自动导入 Neo4j 图谱数据（neo4j-bootstrap）
# docker compose --profile infra up -d --build

# 可选：启动 MCP SSE 服务（需要 MySQL，因此通常与 infra 一起启用）
# docker compose --profile infra --profile mcp up -d --build

# 可选：启动语音识别（FunASR）
# docker compose --profile asr up -d --build
```

访问：
- **Web UI**: http://localhost:3100/
- **API 文档**: http://localhost:3100/api/docs（或直连 http://localhost:5050/docs）

### 📦 数据初始化（首次运行）

当你使用 `--profile infra` 启动时，会自动导入 Neo4j 图谱数据（由 `neo4j-bootstrap` 完成），数据来源：

- `resources/data/kg_data/entities.json`
- `resources/data/kg_data/relations.json`

如需强制重导（危险：会清空 Neo4j 数据库）：

```bash
cd docker
docker compose run --rm neo4j-bootstrap python scripts/import_graph.py --wait-seconds 120 --force --reset
```

> 可选：MySQL 地图数据导入（地图功能需要时再执行）。
> 由于环境差异较大，这一步不默认自动执行，避免影响“一键启动”。

```bash
cd docker
docker compose exec api python scripts/import_pokemon_map.py
```

如果你想彻底清空数据（Neo4j/Milvus/MySQL 等）重新来一遍（需要你曾用 `--profile infra` 启动过）：

```bash
cd docker
docker compose down
# 这些是 bind mount 的数据目录（不属于 named volumes），需要手动删除（Linux/macOS/WSL）
rm -rf volumes/neo4j/data volumes/neo4j/logs volumes/milvus volumes/mysql/data
docker compose --profile infra up -d --build
```

Windows PowerShell：

```powershell
cd docker
docker compose down
Remove-Item -Recurse -Force .\\volumes\\neo4j\\data, .\\volumes\\neo4j\\logs, .\\volumes\\milvus, .\\volumes\\mysql\\data
docker compose --profile infra up -d --build
```

### ✅ 启动验证

- Web UI: http://localhost:3100/
- API Ready: http://localhost:5050/readyz
- Neo4j Browser: http://localhost:7474/（仅当使用 `--profile infra` 启动）

也可以用命令检查：

```bash
cd docker
docker compose ps
docker compose exec -T neo4j cypher-shell 'MATCH (n) RETURN count(n) AS nodes;'  # 仅当使用 --profile infra 启动
```

### 🤝 开发/贡献

本仓库以 Docker 方式运行与复现为主；开发规范、测试与贡献流程见 `CONTRIBUTING.md`。

---

## 🔭 参考项目

- https://github.com/xerrors/Yuxi-Know
- https://github.com/BinNong/meet-libai

  

---

## 📄 License

本项目遵循 **MIT License**，可自由用于商业或个人项目。二次开发请保留原作者与来源信息。
