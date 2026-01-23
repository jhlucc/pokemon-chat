
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

[中文](./README.md) | 📘 English

<img src="resources/picture/11.png" alt="Kemeng Logo" width="27%" />

# 「Kemeng」 <img src="resources/picture/brain-removebg-preview.png" alt="Brain Icon" width="11%" /> Domain Chat Assistant Based on Knowledge Graph and Corpus

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=ffffff)
![LangGraph](https://img.shields.io/badge/LangGraph-Flow-green?style=flat&logo=databricks)
![GraphRAG](https://img.shields.io/badge/GraphRAG-KG-blueviolet?style=flat)
![Agent](https://img.shields.io/badge/Agent-System-orange?style=flat)
![Vue3](https://img.shields.io/badge/Vue-3.0-4FC08D?style=flat&logo=vue.js)
![License](https://img.shields.io/github/license/bitcookies/winrar-keygen.svg?logo=github)

<img src="resources/picture/img.png" alt="System Overview" style="width: 90%;" />

---

## 📝 Project Introduction

Pokémon is one of the most influential IPs worldwide, with a massive universe and character data. Its long-term accumulation across games, animations, cards, and films has resulted in a highly structured knowledge system, making it ideal for knowledge graph modeling and intelligent Q&A scenarios.

With the advancement of LLMs and knowledge-enhanced techniques, building a **multimodal, structured, and interactive AI system** based on the Pokémon universe is now feasible. This project builds a Pokémon knowledge graph using data from Baidu Tieba and Wikipedia, covering characters, attributes, skills, regions, evolution paths, and more. Combined with LLM capabilities, we created a **Pokémon-domain smart assistant** — “Kemeng.”

By integrating **LangGraph pipeline orchestration**, **GraphRAG enhanced retrieval**, and **graph visualization**, users can both get accurate answers through natural language queries and visually explore the Pokémon world. The system also supports geographic mapping, linking Pokémon world locations to real-world coordinates for **spatial visualization**.

This project is designed to be a **transferable, scalable domain assistant template**, making it easy to adapt for other characters or fields (e.g., Su Shi, finance, e-government) by simply changing the knowledge source and graph structure.

---

## 🚀 New Features

- **LangChain & LangGraph**: Supports LangChain 1.x and LangGraph 1.0 for multi-agent orchestration
- **LightRAG Integration**: Integrated with HKU-DS LightRAG for efficient retrieval
- **Advanced RAG**: Features Self-RAG, CRAG, HyDE, and Query Decomposition
- **Agentic Memory**: Long-term memory with user preference adaptation
- **MCP Service**: Supports Model Context Protocol for real-world location mapping
- **Performance**: Built-in Semantic Cache and Speculative RAG for speed

---

## 🎯 System Architecture

The project includes a complete Vue3 + FastAPI stack and a functional Pokémon knowledge graph-based Q&A system. It combines semantic modeling (BERT + TF-IDF + rule matching) with generative Q&A, supporting questions about evolution, attribute restraints, skills, and geographic distribution.

**Core Architecture**:
- **Hybrid Retrieval**: Vector Retrieval (Milvus) + Graph Retrieval (Neo4j) + Keyword Retrieval (BM25)
- **Agent Orchestration**: LangGraph state machine for complex task management
- **Knowledge Enhancement**: GraphRAG for entity relationship extraction

Architecture overview:

<img src="resources/picture/now.png" alt="Architecture" style="width: 100%;" />

## 🎯 Highlights

1. Fine-tuned a Pokémon-domain LLM ("[Kemeng](https://huggingface.co/qwqqwq/qwen2.5-14b-instruct-pokemon-int4)") using web-scraped data.
2. Built a Pokémon knowledge graph based on Wikipedia and forums.
3. Automated NER training with RoBERTa + TF-IDF + rule-based matching.
4. Integrated FunASR (Alibaba DAMO Academy) for ASR (speech-to-text) capabilities.
5. **[NEW]** Implemented **MCP Service** to support mapping and querying of Pokémon world locations to real-world coordinates.
6. Extracted documents with DeepDoc to enhance knowledge base parsing.
7. **[NEW]** Used **LangGraph** to implement multi-agent collaboration (RAG + Search + Graph + MCP).
8. Encapsulated agent base class for multi-agent workflows.
9. Supports graph search, web search, knowledge base search, MCP queries, and voice input, in any combination.

---

## 🛠️ Deployment

> **Requirements**: Docker & Docker Compose

### 🐳 Docker Compose One-Click Start (Recommended)

No manual environment configuration needed. Directly use Docker Compose to start all services:

```bash
# 1. Clone the repository
git clone https://github.com/skygazer42/pokemon-chat.git
cd pokemon-chat

# 2. Configure environment variables (Optional, defaults work out-of-the-box)
cp .env.template .env
# Edit .env file, fill in LLM API KEY (e.g., SILICONFLOW_API_KEY)

# 3. Start all services (API + Web + DB + MCP)
cd docker
docker compose --profile infra --profile mcp up -d --build
```

Access:
- **Web UI**: http://localhost:3100/
- **API Docs**: http://localhost:5050/docs

### 📦 Data Initialization (First Run)

After starting the services, you need to import knowledge base and graph data. Place data files in the `resources` directory ([Download Link](https://pan.baidu.com/s/1o48ankI6l9jaky5MeRqgYw?pwd=rkdy)).

Then execute the import scripts:

```bash
# Enter API container
docker exec -it pokemon-chat-api-1 bash

# Import Neo4j graph data
python scripts/import_graph.py

# Import MySQL map data
python scripts/import_pokemon_map.py
```

### 💻 Local Development Mode

If you wish to run backend/frontend code locally for development:

1. **Start Infrastructure**
   ```bash
   cd docker
   docker compose --profile infra up -d # Starts Neo4j, Milvus, MySQL, FunASR
   ```

2. **Start Backend (Server)**
   ```bash
   cd server
   pip install -r requirements.txt
   python main.py
   ```

3. **Start Frontend (Web)**
   ```bash
   cd web
   npm install
   npm run dev
   ```

---

## 🔭 Reference Projects

- https://github.com/xerrors/Yuxi-Know  
- https://github.com/BinNong/meet-libai  

---

## 📄 License

This project is licensed under the **MIT License**, free for commercial and personal use. Please retain author credits when redistributing.
