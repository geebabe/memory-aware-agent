# Memory-Aware Agent System

A modular, production-grade AI agent framework featuring multi-layered long-term memory and autonomous knowledge ingestion using Oracle Database 23ai Vector Search.

---

## 🚀 Overview

This project implements a **Memory-Aware Agent** capable of searching, learning, and recalling information across multiple dimensions of context. Unlike standard RAG systems, this agent actively manages its own knowledge base by fetching research papers (ArXiv) and web results (Tavily), persisting them into a specialized vector memory.

### Key Features
- **Multi-Layered Memory Architecture**: Separate stores for semantic knowledge, workflows, tools, entities, and summaries.
- **Oracle 23ai Vector Integration**: Leverages high-performance vector search and JSON-duality for robust metadata handling.
- **Autonomous Ingestion**: Tools to automatically search, fetch, and save content to the agent's long-term brain.
- **Thread-Safe Conversational History**: Persistent chat sessions with automatic summary generation.
- **Modular Design**: Decoupled components for agents, memory, tools, and database management.

---

## 🛠 Tech Stack

- **Model**: OpenAI (GPT-4o / GPT-5-mini)
- **Vector Database**: Oracle Database 23ai
- **Framework**: LangChain, LangChain-OracleDB
- **Embeddings**: HuggingFace (`sentence-transformers/paraphrase-mpnet-base-v2`)
- **Search**: Tavily API, ArXiv API

---

## 📂 Project Structure

```text
.
├── app/
│   ├── agent/       # Agent Orchestration & Logic
│   ├── core/        # Configurations & DB Connections
│   ├── memory/      # Memory Managers & Vector Stores
│   ├── tools/       # Tool definitions (Search, Summary, etc.)
│   └── utils/       # Common utility functions
├── legacy/          # Original research code (reference)
├── main.py          # Application entry point
├── verify_refactor.py # System verification script
└── .env             # Environment variables
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- **Python 3.10+**
- **Oracle Database 23ai** (Running locally or via Docker)
- **OpenAI API Key**
- **Tavily API Key** (for web searching)

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
# Optional:
# ORACLE_ADMIN_USER=system
# ORACLE_ADMIN_PASSWORD=YourPassword123
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# If requirements.txt is not present:
pip install openai oracledb langchain-huggingface langchain-community python-dotenv sentence-transformers arxiv
```

### 4. Database Initialization
Run the database setup script once to create the `VECTOR` user and required tablespaces:
```python
from app.core.database import setup_oracle_database
setup_oracle_database(admin_password="YourPassword123")
```

---

## 🏃 Usage

### Verify Installation
Run the verification script to ensure database connectivity and tool registration:
```bash
python verify_refactor.py
```

### Run the Agent
Execute the main application:
```bash
python main.py
```

By default, the agent will:
1. Initialize connection to Oracle.
2. Load the embedding model.
3. Search for research papers if the query requires external knowledge.
4. Auto-save relevant findings to the `SEMANTIC_MEMORY` table.

---

## 🧠 Memory Layers

| Layer | Table Name | Purpose |
| :--- | :--- | :--- |
| **Semantic Knowledge** | `SEMANTIC_MEMORY` | General facts and ingested papers. |
| **Conversational** | `CONVERSATIONAL_MEMORY` | Historical dialogue per thread. |
| **Tool Log** | `TOOL_LOG_MEMORY` | Debug logs of tool calls and results. |
| **Entity** | `ENTITY_MEMORY` | Specific recognized entities (people, orgs). |
| **Workflow** | `WORKFLOW_MEMORY` | Learned procedural steps for tasks. |
| **Summary** | `SUMMARY_MEMORY` | Condensed context of long documents. |

---

## 📜 License
This project is for research and development purposes.
