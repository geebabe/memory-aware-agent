from app.tools.search_tools import register_search_tools
from app.tools.summary_tools import register_summary_tools

def register_all_tools(toolbox, memory_manager, llm_client, knowledge_base_table: str):
    """Register all available tools with the toolbox."""
    search_tools = register_search_tools(toolbox, memory_manager, knowledge_base_table)
    summary_tools = register_summary_tools(toolbox, memory_manager, llm_client)
    return search_tools + summary_tools
