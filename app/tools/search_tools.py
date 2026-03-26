import os
import time
import arxiv
from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults

def register_search_tools(toolbox, memory_manager, knowledge_base_table: str):
    """Register search-related tools with the toolbox."""

    @toolbox.register_tool(augment=True)
    def arxiv_search_candidates(query: str, max_results: int = 5) -> str:
        """Search ArXiv for papers matching a query."""
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for r in search.results():
            results.append(f"Title: {r.title}\nID: {r.entry_id}\nAuthor: {r.authors[0]}\nSummary: {r.summary[:300]}...")
        return "\n\n".join(results) or "No papers found."

    @toolbox.register_tool(augment=True)
    def fetch_and_save_paper_to_kb_db(paper_id: str) -> str:
        """Fetch a full paper from ArXiv and save it to the Knowledge Base."""
        search = arxiv.Search(id_list=[paper_id.split('/')[-1]])
        paper = next(search.results())
        text = f"Title: {paper.title}\nAuthors: {[a.name for a in paper.authors]}\nSummary: {paper.summary}\nURL: {paper.entry_id}"
        memory_manager.write_knowledge_base(text, {"source": "arxiv", "id": paper_id, "title": paper.title})
        return f"Paper '{paper.title}' saved to Knowledge Base."

    @toolbox.register_tool(augment=True)
    def web_search(query: str) -> str:
        """Perform a web search using Tavily."""
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key: return "Error: TAVILY_API_KEY not set."
        search = TavilySearchResults(k=5)
        results = search.run(query)
        return str(results)

    @toolbox.register_tool(augment=True)
    def get_current_time() -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return ["arxiv_search_candidates", "fetch_and_save_paper_to_kb_db", "web_search", "get_current_time"]
