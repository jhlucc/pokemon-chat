<div align="center">

<img src="resources/picture/11.png" alt="图标" width="200" />

# 「可萌」

**基于知识库与知识图谱的专域智能助手**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-005571?style=flat&logo=fastapi)
![Vue](https://img.shields.io/badge/Vue-3.5-4FC08D?style=flat&logo=vue.js)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-green?style=flat)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=ffffff)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

📘 中文 | [English](./README.en.md)

</div>

---

## 核心特性

- **智能体开发**：基于 LangGraph 1.0 的多智能体架构，支持子智能体、工具调用与中间件机制
- **知识库（RAG）**：多格式文档上传，支持 Embedding / Rerank 配置及混合检索
- **知识图谱**：基于 LightRAG 的图谱构建与可视化，支持属性图谱并参与智能体推理
- **平台与工程化**：Vue 3 + FastAPI 架构，支持暗黑模式、Docker 与生产级部署

---

## 你可以用「可萌」做什么？

- 构建 **面向真实业务的 RAG + 知识图谱智能体**
- 将 PDF / Word / Markdown / 图片快速转化为可推理的知识库
- 自动（LightRAG）或手动构建知识图谱，并用于智能体推理
- 使用 LangGraph 1.0 构建多智能体 / 子智能体系统
- 支持知识图谱搜索、网络搜索、知识库搜索、MCP搜索、语音搜索

---

## 最新动态

<details>
<summary>[2026/01] v0.3.0 版本发布</summary>

### 新增
- 优化数据库管理界面，新增玻璃拟态主题样式
- 新增批量导入面板和加载骨架屏
- 更新模型目录，新增多个模型配置
- 优化 UI 组件（标签页、按钮、滑块）

### 优化
- 统一表格样式与主题风格
- 改进输入提示体验

</details>

<details>
<summary>[2025/12] v0.2.0 版本发布</summary>

### 新增
- 完整的 Docker Compose 部署方案
- Neo4j 知识图谱自动导入
- MCP 服务集成
- FunASR 语音识别支持

</details>

---

## 快速开始

克隆代码并初始化：

```bash
git clone https://github.com/skygazer42/pokemon-chat.git
cd pokemon-chat

# 配置环境变量
cp .env.example .env
# 编辑 .env，填写 LLM API KEY
```

使用 Docker 启动项目：

```bash
cd docker

# 默认仅启动 App（API + Web）
docker compose up -d --build

# 可选：启动完整基础设施（Neo4j/Milvus/MySQL）
# docker compose --profile infra up -d --build

# 可选：启动 MCP 服务
# docker compose --profile infra --profile mcp up -d --build
```

等待启动完成后，访问 `http://localhost:3100`

---

## 系统架构

<img src="resources/picture/now.png" alt="架构图" style="width: 100%;" />

架构核心：
- **混合检索**: 向量检索 (Milvus) + 图谱检索 (Neo4j) + 关键词检索 (BM25)
- **智能体编排**: LangGraph 1.0 状态机管理复杂任务流
- **知识增强**: GraphRAG 提取实体关系，支持 Self-RAG、CRAG、HyDE

---

## 项目特色

1. 基于宝可梦数据微调的专域大模型 —— [可萌](https://huggingface.co/qwqqwq/qwen2.5-14b-instruct-pokemon-int4)
2. 基于维基百科构建的宝可梦知识图谱
3. 自动化标注训练 NER 数据（RoBERTa + TF-IDF + 规则匹配）
4. FunASR 语音识别（阿里达摩院开源）
5. MCP 服务支持宝可梦世界与真实世界坐标映射
6. DeepDoc 文档解析增强（抽取自 RAGFlow）
7. LangGraph 多智能体协同（RAG + Search + Graph + MCP）
8. 可迁移的专域智能助手模板系统

---

## 详细部署

> **前置要求**：已安装 Docker / Docker Compose

<details>
<summary>Docker Compose 详细配置</summary>

### 环境变量配置

编辑 `.env` 文件，配置以下选项：

```bash
# LLM 配置
llm_api_key=your_api_key
llm_model_name=Qwen/Qwen2.5-7B-Instruct

# 功能开关
enable_knowledge_graph=true   # Neo4j 知识图谱
enable_knowledge_base=true    # Milvus 知识库
enable_web_search=true        # Web 搜索（需 tavily_api_key）
enable_mcp=true               # MCP 地理服务
enable_asr=true               # 语音识别
```

### 启动配置

```bash
cd docker

# 仅启动 App
docker compose up -d --build

# 启动完整基础设施
docker compose --profile infra up -d --build

# 启动所有服务
docker compose --profile infra --profile mcp --profile asr up -d --build
```

### 数据初始化

使用 `--profile infra` 时会自动导入知识图谱数据。如需手动重导：

```bash
docker compose run --rm neo4j-bootstrap python scripts/import_graph.py --force --reset
```

### 访问地址

- Web UI: http://localhost:3100/
- API 文档: http://localhost:5050/docs
- Neo4j Browser: http://localhost:7474/

</details>

---

## 参考项目

- [Yuxi-Know](https://github.com/xerrors/Yuxi-Know) - 语析知识库系统
- [meet-libai](https://github.com/BinNong/meet-libai)

---

## 参与贡献

欢迎提交 Issue 和 Pull Request！

开发规范与贡献流程见 [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## 📄 License

本项目采用 **MIT License** - 查看 [LICENSE](LICENSE) 文件了解详情。

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个 ⭐️**

[报告问题](https://github.com/skygazer42/pokemon-chat/issues) | [功能请求](https://github.com/skygazer42/pokemon-chat/issues) | [讨论](https://github.com/skygazer42/pokemon-chat/discussions)

</div>
