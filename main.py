import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from app.core.config import load_env, suppress_warnings
from app.core.database import setup_oracle_database, connect_to_oracle
from app.memory.stores import StoreManager, create_conversational_history_table, create_tool_log_table
from app.memory.manager import MemoryManager
from app.tools.base import Toolbox
from app.tools import register_all_tools
from app.agent.orchestrator import AgentOrchestrator

def main():
    suppress_warnings()
    load_env()

    # 1. Database Setup
    # setup_oracle_database() # Run once if needed
    conn = connect_to_oracle(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn="127.0.0.1:1521/FREEPDB1"
    )

    # 2. Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")

    # 3. Memory Setup
    table_names = {
        'knowledge_base': "SEMANTIC_MEMORY",
        'workflow': "WORKFLOW_MEMORY",
        'toolbox': "TOOLBOX_MEMORY",
        'entity': "ENTITY_MEMORY",
        'summary': "SUMMARY_MEMORY",
    }
    conv_table = create_conversational_history_table(conn, "CONVERSATIONAL_MEMORY")
    log_table = create_tool_log_table(conn, "TOOL_LOG_MEMORY")

    store_manager = StoreManager(
        client=conn,
        embedding_function=embedding_model,
        table_names=table_names,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        conversational_table=conv_table,
        tool_log_table=log_table
    )

    memory_manager = MemoryManager(
        conn=conn,
        conversation_table=conv_table,
        knowledge_base_vs=store_manager.get_knowledge_base_store(),
        workflow_vs=store_manager.get_workflow_store(),
        toolbox_vs=store_manager.get_toolbox_store(),
        entity_vs=store_manager.get_entity_store(),
        summary_vs=store_manager.get_summary_store(),
        tool_log_table=log_table
    )

    # 4. Agent and Tools Setup
    llm_client = OpenAI()
    toolbox = Toolbox(memory_manager, llm_client, embedding_model)
    register_all_tools(toolbox, memory_manager, llm_client, table_names['knowledge_base'])

    orchestrator = AgentOrchestrator(memory_manager, toolbox, llm_client)

    # 5. Run Agent
    query = "Find recent papers on LLM memory and explain how they manage long-term context."
    orchestrator.call_agent(query, thread_id="production-test-1")

    conn.close()

if __name__ == "__main__":
    main()
