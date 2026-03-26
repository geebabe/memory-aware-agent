import json as json_lib
from datetime import datetime
from typing import Optional

class MemoryManager:
    """
    A unified memory manager for AI agents using Oracle AI Database.
    Manages 7 types of memory: Conversational, Tool Log, Knowledge Base, Workflow, Toolbox, Entity, Summary.
    """
    def __init__(self, conn, conversation_table: str, knowledge_base_vs, workflow_vs, toolbox_vs,
                 entity_vs, summary_vs, tool_log_table: str | None = None):
        self.conn = conn
        self.conversation_table = conversation_table
        self.knowledge_base_vs = knowledge_base_vs
        self.workflow_vs = workflow_vs
        self.toolbox_vs = toolbox_vs
        self.entity_vs = entity_vs
        self.summary_vs = summary_vs
        self.tool_log_table = tool_log_table

    # ==================== CONVERSATIONAL MEMORY (SQL) ====================
    def write_conversational_memory(self, content: str, role: str, thread_id: str) -> str:
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            id_var = cur.var(str)
            cur.execute(f"INSERT INTO {self.conversation_table} (thread_id, role, content, metadata, timestamp) VALUES (:thread_id, :role, :content, :metadata, CURRENT_TIMESTAMP) RETURNING id INTO :id",
                        {"thread_id": thread_id, "role": role, "content": content, "metadata": "{}", "id": id_var})
            record_id = id_var.getvalue()[0] if id_var.getvalue() else None
        self.conn.commit()
        return record_id

    def read_conversational_memory(self, thread_id: str, limit: int = 10) -> str:
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT role, content, timestamp FROM {self.conversation_table} WHERE thread_id = :thread_id AND summary_id IS NULL ORDER BY timestamp ASC FETCH FIRST :limit ROWS ONLY",
                        {"thread_id": thread_id, "limit": limit})
            results = cur.fetchall()
        messages = [f"[{ts.strftime('%H:%M:%S')}] [{role}] {content}" for role, content, ts in results]
        messages_formatted = '\n'.join(messages) or "(No unsummarized messages found for this thread.)"
        return f"## Conversation Memory\n### What this memory is\nChronological, unsummarized messages from the current thread.\n### Retrieved messages\n\n{messages_formatted}"

    def mark_as_summarized(self, thread_id: str, summary_id: str):
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"UPDATE {self.conversation_table} SET summary_id = :summary_id WHERE thread_id = :thread_id AND summary_id IS NULL",
                        {"summary_id": summary_id, "thread_id": thread_id})
        self.conn.commit()

    # ==================== TOOL LOG MEMORY (SQL) ====================
    def write_tool_log(self, thread_id: str, tool_name: str, tool_args, result: str, status: str = "success",
                       tool_call_id: str | None = None, error_message: str | None = None, metadata: dict | None = None) -> str | None:
        if not self.tool_log_table: return None
        thread_id = str(thread_id)
        tool_args_str = json_lib.dumps(tool_args, ensure_ascii=False) if isinstance(tool_args, (dict, list)) else str(tool_args or "")
        result_str = str(result or "")
        preview = result_str.encode("utf-8")[:2000].decode("utf-8", errors="ignore")
        metadata_str = json_lib.dumps(metadata, ensure_ascii=False) if metadata else "{}"
        with self.conn.cursor() as cur:
            id_var = cur.var(str)
            cur.execute(f"INSERT INTO {self.tool_log_table} (thread_id, tool_call_id, tool_name, tool_args, result, result_preview, status, error_message, metadata, timestamp) VALUES (:thread_id, :tool_call_id, :tool_name, :tool_args, :result, :result_preview, :status, :error_message, :metadata, CURRENT_TIMESTAMP) RETURNING id INTO :id",
                        {"thread_id": thread_id, "tool_call_id": tool_call_id, "tool_name": tool_name, "tool_args": tool_args_str, "result": result_str, "result_preview": preview, "status": status, "error_message": error_message, "metadata": metadata_str, "id": id_var})
            log_id = id_var.getvalue()[0] if id_var.getvalue() else None
        self.conn.commit()
        return log_id

    def read_tool_logs(self, thread_id: str, limit: int = 20) -> list[dict]:
        if not self.tool_log_table: return []
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, tool_call_id, tool_name, tool_args, result_preview, status, error_message, metadata, timestamp FROM {self.tool_log_table} WHERE thread_id = :thread_id ORDER BY timestamp DESC FETCH FIRST :limit ROWS ONLY",
                        {"thread_id": thread_id, "limit": limit})
            rows = cur.fetchall()
        return [{"id": r[0], "tool_call_id": r[1], "tool_name": r[2], "tool_args": r[3], "result_preview": r[4], "status": r[5], "error_message": r[6], "metadata": r[7], "timestamp": r[8].isoformat() if r[8] else None} for r in rows]

    # ==================== VECTOR STORES (Knowledge, Workflow, Toolbox, Entity, Summary) ====================
    def write_knowledge_base(self, text: str | list[str], metadata: dict | list[dict]):
        if isinstance(text, list):
            self.knowledge_base_vs.add_texts([str(t) for t in text], metadata if isinstance(metadata, list) else [metadata]*len(text))
        else:
            self.knowledge_base_vs.add_texts([str(text)], [metadata if isinstance(metadata, dict) else {}])

    def read_knowledge_base(self, query: str, k: int = 3) -> str:
        results = self.knowledge_base_vs.similarity_search(query, k=k)
        content = "\n".join([doc.page_content for doc in results]) or "(No relevant knowledge base passages found.)"
        return f"## Knowledge Base Memory\n### Retrieved passages\n\n{content}"

    def write_workflow(self, query: str, steps: list, final_answer: str, success: bool = True):
        steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        self.workflow_vs.add_texts([f"Query: {query}\nSteps:\n{steps_text}\nAnswer: {final_answer[:200]}"],
                                   [{"query": query, "success": success, "num_steps": len(steps), "timestamp": datetime.now().isoformat()}])

    def read_workflow(self, query: str, k: int = 3) -> str:
        results = self.workflow_vs.similarity_search(query, k=k, filter={"num_steps": {"$gt": 0}})
        content = "\n---\n".join([doc.page_content for doc in results]) or "(No relevant workflows found.)"
        return f"## Workflow Memory\n### Retrieved workflows\n\n{content}"

    def write_toolbox(self, text: str, metadata: dict):
        self.toolbox_vs.add_texts([text], [metadata])

    def read_toolbox(self, query: str, k: int = 3) -> list[dict]:
        results = self.toolbox_vs.similarity_search(query, k=k)
        tools, seen = [], set()
        for doc in results:
            meta = doc.metadata
            name = meta.get("name", "tool")
            if name in seen: continue
            seen.add(name)
            props = {p: {"type": {"<class 'str'>":"string", "str":"string", "<class 'int'>":"integer", "int":"integer", "<class 'float'>":"number", "float":"number", "<class 'bool'>":"boolean", "bool":"boolean"}.get(i.get("type", "string"), "string")} for p, i in meta.get("parameters", {}).items()}
            req = [p for p, i in meta.get("parameters", {}).items() if "default" not in i]
            tools.append({"type": "function", "function": {"name": name, "description": meta.get("description", ""), "parameters": {"type": "object", "properties": props, "required": req}}})
        return tools

    def write_entity(self, name: str, entity_type: str, description: str):
        self.entity_vs.add_texts([f"{name} ({entity_type}): {description}"], [{"name": name, "type": entity_type, "description": description}])

    def read_entity(self, query: str, k: int = 5) -> str:
        results = self.entity_vs.similarity_search(query, k=k)
        entities = [f"• {doc.metadata.get('name', '?')}: {doc.metadata.get('description', '')}" for doc in results]
        entities_formatted = '\n'.join(entities) or "(No entities found.)"
        return f"## Entity Memory\n### Retrieved entities\n\n{entities_formatted}"

    def write_summary(self, summary_id: str, full_content: str, summary: str, description: str, thread_id: str | None = None):
        meta = {"id": summary_id, "full_content": full_content, "summary": summary, "description": description}
        if thread_id: meta["thread_id"] = str(thread_id)
        self.summary_vs.add_texts([f"{summary_id}: {description}"], [meta])
        return summary_id

    def read_summary_memory(self, summary_id: str, thread_id: str | None = None) -> str:
        filters = {"id": summary_id}
        if thread_id: filters["thread_id"] = str(thread_id)
        results = self.summary_vs.similarity_search(summary_id, k=1, filter=filters)
        return results[0].metadata.get('summary', 'No summary content.') if results else f"Summary {summary_id} not found."

    def read_summary_context(self, query: str = "", k: int = 10, thread_id: str | None = None) -> str:
        filters = {"thread_id": str(thread_id)} if thread_id else None
        results = self.summary_vs.similarity_search(query or "summary", k=k, filter=filters)
        lines = ["## Summary Memory", "### Available summaries", "Use expand_summary(id) to retrieve the detailed underlying conversation."]
        if not results: lines.append("(No summaries available.)")
        else:
            for doc in results: lines.append(f"  • [ID: {doc.metadata.get('id', '?')}] {doc.metadata.get('description', 'No description')}")
        return "\n".join(lines)

    def read_conversations_by_summary_id(self, summary_id: str) -> str:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT role, content, timestamp FROM {self.conversation_table} WHERE summary_id = :summary_id ORDER BY timestamp ASC", {"summary_id": summary_id})
            results = cur.fetchall()
        if not results: return f"No conversations found for summary_id: {summary_id}"
        lines = [f"## Expanded Conversations for Summary ID: {summary_id}", f"Total messages: {len(results)}\n"]
        for role, content, timestamp in results:
            ts = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
            lines.extend([f"[{ts}] [{role.upper()}]", content, ""])
        return "\n".join(lines)
