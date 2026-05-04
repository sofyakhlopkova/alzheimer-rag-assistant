# src/rag/pipeline.py
import logging
from typing import List, Dict, Optional, Tuple

from src.rag.context_builder import build_context, build_prompt, build_prompt_with_history
from src.llm.llm_client import query_llm, check_llm_available

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def rag_answer(
    query: str,
    collection,
    embedding_model,
    top_k: int = 5,
    filter_by_year: Optional[Tuple[int, int]] = None,
    filter_full_text_only: bool = False,
    use_llm: bool = True,
    conversation_history: Optional[List[Dict]] = None
) -> Dict:
    """полный RAG ответ на вопрос исследователя"""
    from src.database.vector_store import search_similar, format_results
    
    logging.info(f"RAG запрос: {query}")
    
    # поиск похожих статей
    results = search_similar(
        query=query,
        collection=collection,
        embedding_model=embedding_model,
        n_results=top_k,
        filter_by_year=filter_by_year,
        filter_full_text_only=filter_full_text_only
    )
    
    if not results['ids'][0]:
        return {
            'answer': "не найдено релевантных статей по вашему запросу",
            'sources': [],
            'has_sources': False
        }
    
    # форматирование результатов для отображения
    sources_df = format_results(results)
    
    # контекст и промпт
    context = build_context(results, max_chunks=top_k)
    
    if use_llm and check_llm_available():
        # ответ через LLM
        if conversation_history:
            prompt = build_prompt_with_history(query, context, conversation_history)
        else:
            prompt = build_prompt(query, context)
        
        answer = query_llm(prompt)
        
        if answer is None:
            # fallback
            answer = _build_fallback_answer(results, sources_df)
    else:
        # fallback ответ на основе найденных статей
        answer = _build_fallback_answer(results, sources_df)
    
    # результат
    return {
        'answer': answer,
        'sources': sources_df.to_dict('records'),
        'has_sources': True
    }


def _build_fallback_answer(results: dict, sources_df) -> str:
    """ответ без LLM на основе найденных статей"""
    answer = f"найдено {len(sources_df)} релевантных статей:\n\n"
    
    for _, row in sources_df.iterrows():
        answer += f" {row['title']} ({row['year']})\n"
        answer += f"   релевантность: {row['similarity']:.3f}\n"
        answer += f"   PMID: {row['pmid']}\n"
        answer += f"   фрагмент: {row['text_preview']}\n\n"
    
    return answer