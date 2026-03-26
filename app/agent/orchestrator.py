import json as json_lib
import logging
from app.core.config import MODEL_TOKEN_LIMITS
from app.agent.context import calculate_context_usage, offload_to_summary

AGENT_SYSTEM_PROMPT = """
# Role
You are a memory-aware agentic research assistant with access to tools.

# Context Window Structure (Partitioned Segments)
The user input is a partitioned context window. It contains a `# Question` section followed by memory segments.
Treat each segment as a distinct memory store with a specific purpose:
- `## Conversation Memory`
- `## Knowledge Base Memory`
- `## Workflow Memory`
- `## Entity Memory`
- `## Summary Memory`

# Memory Store Semantics
- Conversation Memory: Recent thread-level dialogue.
- Knowledge Base Memory: Retrieved documents/passages.
- Workflow Memory: Prior execution patterns.
- Entity Memory: Named people/orgs/systems.
- Summary Memory: Compressed older context.

# Summary Expansion Policy
If critical detail is missing or ambiguous in Summary Memory, call `expand_summary(summary_id)`.

# Operating Rules
1. Start with the provided memory segments before using tools.
2. If memory is insufficient, state what is missing and then use an appropriate tool.
3. For conversation compaction, use `summarize_and_store` with `thread_id`.
"""

class AgentOrchestrator:
    def __init__(self, memory_manager, toolbox, llm_client, model: str = "gpt-5-mini"):
        self.memory_manager = memory_manager
        self.toolbox = toolbox
        self.llm_client = llm_client
        self.model = model

    def execute_tool(self, tool_name: str, tool_args: dict, current_thread_id: str | None = None) -> str:
        """Execute a tool from the toolbox."""
        if tool_name not in self.toolbox._tools_by_name:
            return f"Error: Tool '{tool_name}' not found"

        args = dict(tool_args or {})
        if tool_name == "summarize_and_store" and "thread_id" not in args and current_thread_id:
            args["thread_id"] = str(current_thread_id)

        try:
            return str(self.toolbox._tools_by_name[tool_name](**args) or "Done")
        except Exception as e:
            return f"Error executing tool {tool_name}: {e}"

    def call_agent(self, query: str, thread_id: str = "1", max_iterations: int = 10) -> str:
        """Main agent loop with memory and context management."""
        thread_id = str(thread_id)
        print(f"\n🧠 BUILDING CONTEXT (Thread: {thread_id})...")

        # 1. Build context from memory
        memory_context = ""
        memory_context += self.memory_manager.read_conversational_memory(thread_id) + "\n\n"
        memory_context += self.memory_manager.read_knowledge_base(query) + "\n\n"
        memory_context += self.memory_manager.read_workflow(query) + "\n\n"
        memory_context += self.memory_manager.read_entity(query) + "\n\n"
        memory_context += self.memory_manager.read_summary_context(query, thread_id=thread_id) + "\n\n"

        # 2. Manage context window
        usage = calculate_context_usage(memory_context, self.model)
        if usage['percent'] > 80:
            print(f"⚠️ Context usage {usage['percent']}% - offloading...")
            memory_context, _ = offload_to_summary(memory_context, self.memory_manager, self.llm_client, thread_id=thread_id)

        # 3. Retrieve tools
        dynamic_tools = self.memory_manager.read_toolbox(query, k=5)
        print(f"🔧 Tools retrieved: {[t['function']['name'] for t in dynamic_tools]}")

        # 4. Initialize messages
        full_context = f"# Question\n{query}\n\n{memory_context}"
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": full_context}
        ]

        # 5. Persist user message and extract entities
        self.memory_manager.write_conversational_memory(query, "user", thread_id)
        # Entity extraction logic could be added here

        steps = []
        print("\n🤖 AGENT RUNNING...")
        for iteration in range(max_iterations):
            print(f"--- Iteration {iteration + 1} ---")
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=dynamic_tools if dynamic_tools else None,
                tool_choice="auto" if dynamic_tools else None
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    print(f"🛠️ Tool call: {tc.function.name}")
                    result = self.execute_tool(tc.function.name, json_lib.loads(tc.function.arguments), thread_id)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})
                    self.memory_manager.write_tool_log(thread_id, tc.function.name, tc.function.arguments, result)
                    steps.append(f"{tc.function.name} → {result[:100]}...")
            else:
                final_answer = msg.content
                print(f"✅ Final Answer: {final_answer[:100]}...")
                self.memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)
                self.memory_manager.write_workflow(query, steps, final_answer)
                return final_answer

        return "Agent reached maximum iterations without a final answer."
