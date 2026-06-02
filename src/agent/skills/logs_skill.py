from langchain.tools import tool
from src.agent.skills.registry import AgentSkill, registry
from collections import deque
import os

@tool
def get_logs(qtd_linha: int = 50, level: str | None = None, busca: str | None = None):
    """Busca e retorna as últimas linhas dos logs salvos a partir de 'rag.log'.
    
    Parâmetros:
    - qtd_linha: quantidade máxima de linhas a retornar (padrão 50, evite valores muito altos para não estourar limite de tokens).
    - level: opcional, filtra por nível do log (ex: 'ERROR', 'WARNING', 'INFO').
    - busca: opcional, termo ou palavra-chave para buscar nos logs (ex: 'rate_limit', 'failed').
    """
    log_path = "rag.log"

    if not os.path.exists(log_path):
        return {
            "result": "Arquivo de log não encontrado",
            "final": True
        }
    
    try:
        with open(log_path, "r", encoding="utf-8") as file:
            all_lines = file.readlines()
            
        filtered_lines = []
        for line in all_lines:
            line_str = line.strip()[:500]
            if not line_str:
                continue
            
            # Filtro por level (ex: INFO, WARNING, ERROR)
            if level and level.upper() not in line_str.upper():
                continue
                
            # Filtro por termo de busca
            if busca and busca.lower() not in line_str.lower():
                continue
                
            filtered_lines.append(line_str)
            
        # Pega as últimas 'qtd_linha' linhas filtradas
        last_lines = filtered_lines[-qtd_linha:] if qtd_linha > 0 else filtered_lines
        
        logs = "\n".join(last_lines)
        
        if not logs:
            return {
                "result": "Nenhum log correspondente aos filtros foi encontrado no arquivo.",
                "final": True
            }
            
        return {
            "result": logs,
            "final": False
        }

    except Exception as e:
        return {
            "result": f"Erro ao ler os logs: {e}",
            "final": False
        }

class LogSkill(AgentSkill):
    name = "Logs"
    description = "Busca os logs salvos das execuções anteriores"

    @property
    def tools(self):
        return [get_logs]
    
    @property
    def prompt_instructions(self):
        return """### Skill: Logs

**get_logs(qtd_linha, level, busca)**: retorna os logs salvos das execuções anteriores do arquivo 'rag.log'.
- `qtd_linha`: Quantidade de linhas a retornar (padrão é 50). Ajuste para valores menores se quiser evitar limites de tokens.
- `level`: Filtro de nível (ex: 'ERROR', 'WARNING', 'INFO'). Sempre prefira filtrar por 'ERROR' ou 'WARNING' se o usuário estiver investigando erros ou falhas.
- `busca`: Termo de busca específico (ex: um nome de erro ou componente).

**Regras:**
- Prefixo da resposta: "**Com base no arquivo de log** \n"
- Se o usuário pedir para analisar erros ou problemas, prefira chamar a ferramenta filtrando por `level='ERROR'` ou `level='WARNING'` primeiro para economizar tokens.
- Se o log for muito grande e falhar por limite de tokens, tente novamente com uma quantidade menor de linhas (`qtd_linha=20` ou `30`) ou filtros mais estritos.
- Formate os logs deixando somente os mais relevantes e em formato estruturado
"""

registry.register(LogSkill())

