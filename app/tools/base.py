import inspect
import uuid
import json as json_lib
from typing import Callable, Optional, Union
from pydantic import BaseModel

class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""
    name: str
    description: str
    signature: str
    parameters: dict
    return_type: str

class Toolbox:
    """Registry for agent-callable tools with semantic retrieval support."""
    def __init__(self, memory_manager, llm_client, embedding_function, model: str = "gpt-5"):
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.embedding_function = embedding_function
        self.model = model
        self._tools_by_name: dict[str, Callable] = {}

    def _get_embedding(self, text: str) -> list[float]:
        if hasattr(self.embedding_function, 'embed_query'):
            return self.embedding_function.embed_query(text)
        elif callable(self.embedding_function):
            return self.embedding_function(text)
        raise ValueError("embedding_function must be callable or have embed_query method")

    def _augment_docstring(self, docstring: str, source_code: str = "") -> str:
        if not docstring.strip() and not source_code.strip(): return "No description provided."
        prompt = f"Analyze the following function and its docstring, then produce a richer, more detailed description for an AI agent. Include a one-line summary, step-by-step functionality, usage scenarios, and caveats.\n\nOriginal docstring:\n{docstring}\n\nSource code:\n```python\n{source_code}\n```\n\nReturn ONLY the improved docstring."
        response = self.llm_client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], max_completion_tokens=2000)
        return response.choices[0].message.content.strip()

    def _generate_queries(self, docstring: str, num_queries: int = 5) -> list[str]:
        prompt = f"Based on this tool description, generate {num_queries} diverse example queries a user might ask when they need this tool. Return ONLY a JSON array of strings.\n\nDescription:\n{docstring}"
        response = self.llm_client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], max_completion_tokens=2000)
        try: return json_lib.loads(response.choices[0].message.content.strip())
        except: return [response.choices[0].message.content.strip()]

    def _get_tool_metadata(self, func: Callable) -> ToolMetadata:
        sig = inspect.signature(func)
        params = {n: {"name": n, "type": str(p.annotation) if p.annotation != inspect.Parameter.empty else "Any", "default": str(p.default) if p.default != inspect.Parameter.empty else None} for n, p in sig.parameters.items()}
        return ToolMetadata(name=func.__name__, description=func.__doc__ or "No description", signature=str(sig), parameters=params, return_type=str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "Any")

    def register_tool(self, func: Optional[Callable] = None, augment: bool = False) -> Union[str, Callable]:
        def decorator(f: Callable) -> str:
            name = f.__name__
            self._tools_by_name[name] = f
            doc = f.__doc__ or ""
            sig = str(inspect.signature(f))
            meta = self._get_tool_metadata(f)
            if augment:
                try: src = inspect.getsource(f)
                except: src = ""
                aug_doc = self._augment_docstring(doc, src)
                queries = self._generate_queries(aug_doc)
                emb_text = f"{name} {aug_doc} {sig} {' '.join(queries)}"
                meta.description = aug_doc
                tool_dict = {"_id": str(uuid.uuid4()), "embedding": self._get_embedding(emb_text), "queries": queries, "augmented": True, **meta.model_dump()}
            else:
                tool_dict = {"_id": str(uuid.uuid4()), "embedding": self._get_embedding(f"{name} {doc} {sig}"), "augmented": False, **meta.model_dump()}
            self.memory_manager.write_toolbox(f"{name} {doc} {sig}", tool_dict)
            return name
        return decorator(func) if func else decorator
