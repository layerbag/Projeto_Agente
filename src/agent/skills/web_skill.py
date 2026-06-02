from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from src.mlops.metrics import measure_time
from src.mlops.observability import logger
from src.agent.skills.registry import AgentSkill, registry


@tool
@measure_time("web_search")
def web_search(query: str) -> dict:
    """Busca informações atualizadas na web. Use para notícias, preços atuais,
    informações que podem ter mudado desde a última indexação dos documentos, ou caso não tenha a informação nos documentos indexados."""

    search = DuckDuckGoSearchRun()
    result = search.run(query)
    logger.info(f"web_search retornou {len(result)} caracteres para: {query[:80]}")
    return {
        "result": result,
        "final": False
    }


class WebSearchSkill(AgentSkill):
    name = "web"
    description = "Busca informações atualizadas na web (DuckDuckGo)"

    @property
    def tools(self):
        return [web_search]

    @property
    def prompt_instructions(self):
        return """### Skill: web — Busca na Web

**web_search(query)**: Busca informações atuais na web via DuckDuckGo.

**Regras:**
- Use APENAS quando a base de conhecimento local não tiver a resposta ou precisar de dados atualizados
- Prefixo da resposta: "De acordo com a web"."""


registry.register(WebSearchSkill())
