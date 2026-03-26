from app.agent.context import expand_summary, summarize_and_store

def register_summary_tools(toolbox, memory_manager, llm_client):
    """Register summary-related tools with the toolbox."""

    @toolbox.register_tool(augment=True)
    def expand_summary(summary_id: str) -> str:
        """
        Expand a summary reference to retrieve the original conversations.
        Use when you need more details from a [Summary ID: xxx] reference.
        """
        summary_text = memory_manager.read_summary_memory(summary_id)
        original_conversations = memory_manager.read_conversations_by_summary_id(summary_id)
        return f"## Summary Context\n{summary_text}\n\n{original_conversations}"

    @toolbox.register_tool(augment=True)
    def summarize_and_store(text: str, thread_id: str = None) -> str:
        """
        Summarize long text and store in memory.
        If thread_id is provided, summarize unsummarized conversation units from that thread.
        """
        from app.agent.context import summarize_conversation
        if thread_id:
            result = summarize_conversation(thread_id, memory_manager, llm_client)
            if result.get("status") == "nothing_to_summarize":
                return f"No unsummarized messages found for thread {thread_id}."
            return f"Thread {thread_id} summarized. Summary ID: {result['id']}"
        else:
            from app.agent.context import summarise_context_window
            result = summarise_context_window(text, memory_manager, llm_client)
            return f"Text summarized. Summary ID: {result['id']}"

    return ["expand_summary", "summarize_and_store"]
