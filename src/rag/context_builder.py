import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config.settings import config


def build_context(search_results: dict, max_chunks: int = config.MAX_CHUNKS, max_text_length: int = config.MAX_TEXT_LENGTH) -> str:
    """собирает контекст из результатов поиска для передачи в LLM"""
    context_parts = []
    
    for i in range(min(len(search_results['ids'][0]), max_chunks)):
        metadata = search_results['metadatas'][0][i]
        document = search_results['documents'][0][i]
        distance = search_results['distances'][0][i]
        
        # конвертируем расстояние в score 
        score = 1 - (distance / 2)
        
        # заголовок источника
        title = metadata.get('title', 'без названия')
        year = metadata.get('year', 'неизвестно')
        pmid = metadata.get('pmid', '')
        
        # обрезаем текст
        if len(document) > max_text_length:
            document = document[:max_text_length] + "..."
        
        context_part = f"""{title} ({year}) | PMID: {pmid} | Релевантность: {score:.3f}
{document}
"""
        context_parts.append(context_part)
    
    context = "\n---\n".join(context_parts)
    logging.info(f"построен контекст из {len(context_parts)} источников")
    
    return context


def get_system_prompt() -> str:
    """возвращает системный промпт для LLM"""
    return """You are a scientific research assistant specializing in Alzheimer's disease.

RULES:
- Respond ONLY in English using Latin alphabet
- Base answer on provided context
- Do not invent information. Aggregate information from sources when possible
- If context lacks specific answer, summarize what sources collectively indicate
- Say "Limited information found" if some relevant data exists
- Do not use phrases like "based on the provided context" or "the sources mention"
- Do not include section headers in your response

FORMAT:
1. Summary: [2-3 sentence overview of key findings]
2. Key findings: [bullet points with key information]
3. Details: [elaboration on most important points, cite sources]
4. Limitations: [if applicable, note gaps in current knowledge]

Always cite sources using PMID when referencing specific findings."""


def build_prompt(query: str, context: str, system_prompt: str = None) -> str:
    """строит промпт для LLM"""
    if system_prompt is None:
        system_prompt = get_system_prompt()
    
    return f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def build_prompt_with_history(query: str, context: str, history: List[Dict]) -> str:
    """построение промпта с учётом истории диалога"""
    history_text = ""
    for msg in history[-5:]:
        role = "Researcher" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    
    return f"""{get_system_prompt()}

CONVERSATION HISTORY:
{history_text}

CURRENT CONTEXT:
{context}

CURRENT QUESTION:
{query}

ANSWER:"""