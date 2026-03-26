import os
import sys
import oracledb
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Add current directory to path
sys.path.append(os.getcwd())

from app.core.config import load_env, suppress_warnings
from app.core.database import connect_to_oracle, table_exists
from app.memory.stores import StoreManager, create_conversational_history_table, create_tool_log_table
from app.memory.manager import MemoryManager
from app.tools.base import Toolbox
from app.tools import register_all_tools
from app.agent.orchestrator import AgentOrchestrator

def test_refactoring():
    suppress_warnings()
    load_env()

    print("--- Testing Database Connection ---")
    conn = connect_to_oracle(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn="127.0.0.1:1521/FREEPDB1"
    )
    assert conn is not None
    print("✅ Database connected.")

    print("\n--- Testing Tables Creation ---")
    conv_table = create_conversational_history_table(conn, "TEST_CONV_MEM")
    log_table = create_tool_log_table(conn, "TEST_LOG_MEM")
    assert table_exists(conn, "TEST_CONV_MEM")
    assert table_exists(conn, "TEST_LOG_MEM")
    print("✅ SQL tables verified.")

    print("\n--- Testing MemoryManager ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    table_names = {
        'knowledge_base': "TEST_KB",
        'workflow': "TEST_WF",
        'toolbox': "TEST_TB",
        'entity': "TEST_ENT",
        'summary': "TEST_SUM",
    }
    store_manager = StoreManager(
        client=conn,
        embedding_function=embedding_model,
        table_names=table_names,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        conversational_table=conv_table,
        tool_log_table=log_table
    )
    mm = MemoryManager(
        conn=conn,
        conversation_table=conv_table,
        knowledge_base_vs=store_manager.get_knowledge_base_store(),
        workflow_vs=store_manager.get_workflow_store(),
        toolbox_vs=store_manager.get_toolbox_store(),
        entity_vs=store_manager.get_entity_store(),
        summary_vs=store_manager.get_summary_store(),
        tool_log_table=log_table
    )

    mm.write_conversational_memory("Hello", "user", "test-thread")
    history = mm.read_conversational_memory("test-thread")
    assert "Hello" in history
    print("✅ MemoryManager read/write verified.")

    print("\n--- Testing Toolbox and Tools ---")
    llm_client = OpenAI()
    toolbox = Toolbox(mm, llm_client, embedding_model)
    tools = register_all_tools(toolbox, mm, llm_client, "TEST_KB")
    assert len(tools) > 0
    print(f"✅ Registered {len(tools)} tools.")

    print("\n--- Testing Agent Orchestrator ---")
    orchestrator = AgentOrchestrator(mm, toolbox, llm_client)
    # Perform a simple logic test (without full LLM loop to save time/tokens)
    assert orchestrator.model == "gpt-5-mini"
    print("✅ Orchestrator initialized.")

    # Cleanup test tables
    with conn.cursor() as cur:
        for t in ["TEST_CONV_MEM", "TEST_LOG_MEM", "TEST_KB", "TEST_WF", "TEST_TB", "TEST_ENT", "TEST_SUM"]:
            try: cur.execute(f"DROP TABLE {t} PURGE")
            except: pass
    conn.commit()
    conn.close()
    print("\n✅ Verification SUCCESS!")

if __name__ == "__main__":
    test_refactoring()
