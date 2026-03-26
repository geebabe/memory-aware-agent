import oracledb
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.retrievers.hybrid_search import OracleVectorizerPreference
from app.core.database import table_exists

def create_conversational_history_table(conn, table_name: str = "CONVERSATIONAL_MEMORY"):
    """Create a table to store conversational history."""
    if table_exists(conn, table_name):
        return table_name
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                thread_id VARCHAR2(100) NOT NULL,
                role VARCHAR2(50) NOT NULL,
                content CLOB NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata CLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary_id VARCHAR2(100) DEFAULT NULL
            )
        """)
        cur.execute(f"CREATE INDEX idx_{table_name.lower()}_thread_id ON {table_name}(thread_id)")
        cur.execute(f"CREATE INDEX idx_{table_name.lower()}_timestamp ON {table_name}(timestamp)")
    conn.commit()
    return table_name

def create_tool_log_table(conn, table_name: str = "TOOL_LOG_MEMORY"):
    """Create a table to store raw tool execution logs per thread."""
    if table_exists(conn, table_name):
        return table_name
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                thread_id VARCHAR2(100) NOT NULL,
                tool_call_id VARCHAR2(200),
                tool_name VARCHAR2(200) NOT NULL,
                tool_args CLOB,
                result CLOB,
                result_preview VARCHAR2(2000),
                status VARCHAR2(30) DEFAULT 'success',
                error_message CLOB,
                metadata CLOB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(f"CREATE INDEX idx_{table_name.lower()}_thread_id ON {table_name}(thread_id)")
        cur.execute(f"CREATE INDEX idx_{table_name.lower()}_tool_name ON {table_name}(tool_name)")
        cur.execute(f"CREATE INDEX idx_{table_name.lower()}_timestamp ON {table_name}(timestamp)")
    conn.commit()
    return table_name

class StoreManager:
    """Manages all stores (vector stores and SQL tables)."""
    def __init__(self, client, embedding_function, table_names, distance_strategy, conversational_table,
                 tool_log_table: str | None = None):
        self.client = client
        self.embedding_function = embedding_function
        self.distance_strategy = distance_strategy
        self._conversational_table = conversational_table
        self._tool_log_table = tool_log_table
        self._knowledge_base_vs = OracleVS(client=client, embedding_function=embedding_function, table_name=table_names['knowledge_base'], distance_strategy=distance_strategy)
        self._workflow_vs = OracleVS(client=client, embedding_function=embedding_function, table_name=table_names['workflow'], distance_strategy=distance_strategy)
        self._toolbox_vs = OracleVS(client=client, embedding_function=embedding_function, table_name=table_names['toolbox'], distance_strategy=distance_strategy)
        self._entity_vs = OracleVS(client=client, embedding_function=embedding_function, table_name=table_names['entity'], distance_strategy=distance_strategy)
        self._summary_vs = OracleVS(client=client, embedding_function=embedding_function, table_name=table_names['summary'], distance_strategy=distance_strategy)
        self._kb_vectorizer_pref = None

    def get_conversational_table(self): return self._conversational_table
    def get_tool_log_table(self): return self._tool_log_table
    def get_knowledge_base_store(self): return self._knowledge_base_vs
    def get_workflow_store(self): return self._workflow_vs
    def get_toolbox_store(self): return self._toolbox_vs
    def get_entity_store(self): return self._entity_vs
    def get_summary_store(self): return self._summary_vs

    def setup_hybrid_search(self, preference_name="KB_VECTORIZER_PREF"):
        self._kb_vectorizer_pref = OracleVectorizerPreference.create_preference(vector_store=self._knowledge_base_vs, preference_name=preference_name)
        return self._kb_vectorizer_pref
