# app/streamlit_app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


from src.database.chroma_client import init_chroma_db
from src.database.collection import create_or_get_collection
from src.embedding.model_manager import load_embedding_model
from src.database.vector_store import search_similar, format_results
from src.rag.pipeline import rag_answer
from src.llm.llm_client import check_llm_available

from src.config.settings import config

# настройка страницы
st.set_page_config(
    page_title="Alzheimer RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .similarity-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .similarity-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 0.8rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# инициализация состояния сессии
def init_session_state():
    """инициализация переменных сессии"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'use_llm' not in st.session_state:
        llm_available = check_llm_available()
        print(f"[DEBUG] LLM доступна: {llm_available}")  # отладка
        st.session_state.use_llm = llm_available
    
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

import time
from functools import lru_cache

# глобальная переменная для хранения загруженных компонентов
_COMPONENTS_CACHE = None

# текущий кэш 
@st.cache_resource
def load_rag_components():
    """загрузка компонентов RAG системы (кэшируется)"""
    with st.spinner(" Загрузка системы..."):
        client = init_chroma_db(config.DB_DIRECTORY)
        collection = create_or_get_collection(client, config.COLLECTION_NAME)
        embedding_model = load_embedding_model(config.EMBEDDING_MODEL)
        return collection, embedding_model

@st.cache_resource
def load_eval_model():
    """загрузка модели для оценки RAG"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# дополнительное кэширование для текстовых запросов
@st.cache_data(ttl=600)  # кэш на 10 минут
def get_embeddings_cached(texts: tuple):  # tuple потому что list не хэшируемый
    """кэширование эмбеддингов для текстов"""
    collection, embedding_model = load_rag_components()
    return embedding_model.encode(list(texts)).tolist()

# функция для отладки кэша
def debug_cache_status():
    """отладка состояния кэша"""
    st.sidebar.markdown("###  Статистика")
    
    # Простая проверка - сколько раз вызывалась загрузка
    if 'load_count' not in st.session_state:
        st.session_state.load_count = 0
    
    # Инкрементируем при каждом запуске main
    st.session_state.load_count += 1
    
    st.sidebar.metric("запусков main()", st.session_state.load_count)
    
    # Если load_count > 1, а load_rag_components все еще должна быть закэширована
    if st.session_state.load_count == 1:
        st.sidebar.success(" компоненты загружаются (первый запуск)")
    else:
        st.sidebar.info(" компоненты из кэша (повторный запуск)")


def display_sources(sources_df: pd.DataFrame):
    """отображение источников"""
    st.markdown("** Источники:**")
    
    for idx, row in sources_df.iterrows():
        if row['similarity'] >= 0.7:
            score_class = "similarity-high"
        elif row['similarity'] >= 0.5:
            score_class = "similarity-medium"
        else:
            score_class = ""
        
        st.markdown(f"""
        <div class="source-card" style="font-size:0.85rem;">
            <b>{row['title'][:]}...</b><br>
             {row['year']} |  PMID: {row['pmid']}<br>
            <span class="{score_class}"> Релевантность: {row['similarity']:.3f}</span><br>
             {row['text_preview'][:150]}...
        </div>
        """, unsafe_allow_html=True)


def display_chat_history():
    """отображение истории чата"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message-user">
                <b> вы:</b> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**ассистент:** {message['content']}")
            
            if "sources" in message and message["sources"]:
                with st.expander(" показать источники"):
                    for source in message["sources"][:]:
                        st.markdown(f"- **{source['title'][:]}** (релевантность: {source['similarity']:.3f})")


def get_rag_answer_with_history(query: str, collection, embedding_model, top_k: int = 5, 
                                  filter_by_year=None, filter_full_text_only=False) -> dict:
    """
    получение RAG ответа с учётом истории диалога
    """
    from src.rag.pipeline import rag_answer
    from src.llm.llm_client import check_llm_available
    import streamlit as st
    
    # добавление истории в запрос
    history_context = ""
    if len(st.session_state.messages) > 0:
        recent_messages = st.session_state.messages[-5:]
        history_context = "Previous conversation:\n"
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role}: {msg['content']}\n"
        history_context += "\n"
    
    enhanced_query = f"{history_context}Current question: {query}"

    try:
        ollama_works = check_llm_available()
        print(f"[DEBUG] check_llm_available() = {ollama_works}")
    except Exception as e:
        print(f"[DEBUG] Ошибка check_llm_available: {e}")
        ollama_works = False  
    
    # проверяем настройки пользователя
    use_llm_setting = st.session_state.get('use_llm', True)
    print(f"[DEBUG] use_llm_setting = {use_llm_setting}")
    
    # LLM если доступен
    if ollama_works and use_llm_setting:
        print("[DEBUG]  использую LLM")
        
        rag_result = rag_answer(
            query=enhanced_query,
            collection=collection,
            embedding_model=embedding_model,
            top_k=top_k,
            filter_by_year=filter_by_year,
            filter_full_text_only=filter_full_text_only,
            use_llm=True
        )

        eval_metrics = evaluate_rag_response(
            question=enhanced_query,
            answer=rag_result['answer'],
            contexts=[s.get('text', '') for s in rag_result.get('sources', [])]
        )
        
        # сохранение в session_state
        if 'eval_history' not in st.session_state:
            st.session_state.eval_history = []
        st.session_state.eval_history.append({
            'query': enhanced_query,
            **eval_metrics,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        return {
            'answer': rag_result['answer'],
            'sources': rag_result.get('sources', []),
            'has_sources': len(rag_result.get('sources', [])) > 0,
            'evaluation': eval_metrics 
        }
    else:
        # fallback
        print("[DEBUG]  использую fallback (без LLM)")
        results = search_similar(
            query=enhanced_query,
            collection=collection,
            embedding_model=embedding_model,
            n_results=top_k,
            filter_by_year=filter_by_year,
            filter_full_text_only=filter_full_text_only
        )
        
        if results['ids'][0]:
            sources_df = format_results(results)
            sources = sources_df.to_dict('records')
            
            # ответ без LLM
            answer = f"**найдено {len(sources)} релевантных статей**\n\n"
            for i, source in enumerate(sources[:3], 1):
                answer += f"{i}. **{source['title']}** ({source['year']})\n"
                answer += f"    Релевантность: {source['similarity']:.3f} |  PMID: {source['pmid']}\n"
                answer += f"    {source['text_preview']}\n\n"
            answer += " *для получения сгенерированного ответа, убедитесь что LLM запущена*"
            
            return {
                'answer': answer,
                'sources': sources,
                'has_sources': True
            }
        else:
            return {
                'answer': "к сожалению, по вашему запросу ничего не найдено",
                'sources': [],
                'has_sources': False
            }

def evaluate_rag_response(question: str, answer: str, contexts: List[str]) -> Dict:
    """
    оценка RAG ответа с помощью эмбеддингов
    """
    if not answer or not question:
        return {
            'answer_relevancy': 0.0,
            'faithfulness': 0.0,
            'context_precision': 0.0
        }
    
    eval_model = load_eval_model()
    
    # эмбеддинги
    q_emb = eval_model.encode([question])
    a_emb = eval_model.encode([answer])
    
    # релевантность ответа вопросу
    relevancy = cosine_similarity(q_emb, a_emb)[0][0]
    
    # точность контекста и верность фактам
    if contexts:
        context_text = ' '.join(contexts[:3])  # топ-3 источника
        c_emb = eval_model.encode([context_text])
        
        context_precision = cosine_similarity(c_emb, q_emb)[0][0]
        faithfulness = cosine_similarity(a_emb, c_emb)[0][0]
    else:
        faithfulness = 0.0
        context_precision = 0.0
    
    metrics = {
        'answer_relevancy': round(relevancy, 3),
        'faithfulness': round(faithfulness, 3),
        'context_precision': round(context_precision, 3)
    }
    
    # вывод в консоль
    print(f"\n RAG EVALUATION")
    print(f"   answer relevancy: {metrics['answer_relevancy']:.3f}")
    print(f"   faithfulness: {metrics['faithfulness']:.3f}")
    print(f"   context precision: {metrics['context_precision']:.3f}")
    
    return metrics
        
def main():
    # инициализация
    init_session_state()
    
    # заголовок
    st.markdown('<div class="main-header"> Alzheimer RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # боковая панель
    with st.sidebar:
        st.markdown("##  Настройки")
        
        # настройка LLM
        llm_available = check_llm_available()
        if llm_available:
            st.session_state.use_llm = st.checkbox(" использовать генерацию ответов (LLM)", value=st.session_state.use_llm)
        else:
            st.warning(" LLM не доступна, режим поиска без генерации")
            st.session_state.use_llm = False
        
        st.markdown("---")
        st.markdown("##  Параметры поиска")
        top_k = st.slider("количество результатов", min_value=3, max_value=15, value=5)
        filter_year = st.checkbox(" фильтр по годам")
        
        if filter_year:
            col_year1, col_year2 = st.columns(2)
            with col_year1:
                min_year = st.number_input("с", min_value=1990, max_value=2024, value=2020)
            with col_year2:
                max_year = st.number_input("по", min_value=1990, max_value=2024, value=2024)
        else:
            min_year, max_year = None, None
        
        full_text_only = st.checkbox(" только статьи с полным текстом")
        
        st.markdown("---")
        st.markdown("##  Статистика")
        
        try:
            import json
            if Path(config.PATH_STORAGE_INFO).exists():
                with open(config.PATH_STORAGE_INFO, 'r') as f:
                    stats = json.load(f)
                st.metric("всего статей", stats.get('total_articles', '?'))
                st.metric("с полным текстом", stats.get('articles_with_full_text', '?'))
        except:
            st.info("Статистика временно недоступна")

        st.markdown("---")
        st.markdown("##  RAG качество")

        if st.button(" показать метрики последнего ответа"):
            if 'eval_history' in st.session_state and st.session_state.eval_history:
                last = st.session_state.eval_history[-1]
                
                cols = st.columns(3)
                
                with cols[0]:
                    st.metric("answer relevancy", f"{last['answer_relevancy']:.2f}")
                
                with cols[1]:
                    st.metric("faithfulness", f"{last['faithfulness']:.2f}")
                
                with cols[2]:
                    st.metric("context precision", f"{last['context_precision']:.2f}")
                
                # общая оценка
                avg_score = (last['answer_relevancy'] + last['faithfulness'] + last['context_precision']) / 3
                st.progress(float(avg_score))
                
                if avg_score >= 0.7:
                    st.success(" отличное качество ответа")
                elif avg_score >= 0.5:
                    st.info(" хорошее качество")
                else:
                    st.warning(" качество ниже среднего, требуется доработка")
            else:
                st.info("задайте вопрос, чтобы увидеть оценку")

        # история оценок
        if st.checkbox(" показать историю оценок"):
            if 'eval_history' in st.session_state and st.session_state.eval_history:
                history_df = pd.DataFrame(st.session_state.eval_history)
                st.dataframe(
                    history_df[['query', 'answer_relevancy', 'faithfulness', 'context_precision']].tail(5),
                    use_container_width=True
                )
            else:
                st.info("история пуста")
        
        st.markdown("---")
        st.markdown("##  Примеры запросов")
        
        example_queries = [
            "What are potential drug targets for Alzheimer's?",
            "Are the targets druggable with small molecules?",
            "What additional studies are needed for tau-targeting therapies?",
            "Which anti-inflammatory targets are promising?"
        ]
        
        for q in example_queries:
            if st.button(q, key=q[:30]):
                st.session_state['query_input'] = q 
                st.session_state['auto_query'] = True  
                st.rerun()
        
        # кнопка очистки чата
        st.markdown("---")
        if st.button(" очистить историю чата", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()
    
    # основная область - чат
    st.markdown("##  Диалог с ассистентом")
    
    # отображение истории чата
    display_chat_history()
    
    # загрузка компонентов RAG
    collection, embedding_model = load_rag_components()
    
    # поле ввода вопроса
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                " Ваш вопрос:",
                value=st.session_state.get('query_input', ''),
                key="user_input",
                placeholder="Например: What are the most promising druggable targets for Alzheimer's disease?",
                label_visibility="collapsed"
            )
        with col2:
            # Проверяем, нужно ли автоматически отправить
            auto_send = st.session_state.get('auto_query', False)
            if auto_send:
                st.session_state.auto_query = False
                send_button = True
            else:
                send_button = st.button(" Отправить", type="primary", use_container_width=True)
    
    # обработка вопроса
    if send_button and query:
        # добавление вопроса пользователя в историю
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state['query'] = ""  # очищение сохраненного запроса
        
        # ответ
        with st.spinner(" Анализирую статьи и генерирую ответ..."):
            year_filter = (min_year, max_year) if filter_year and min_year and max_year else None
            
            result = get_rag_answer_with_history(
                query=query,
                collection=collection,
                embedding_model=embedding_model,
                top_k=top_k,
                filter_by_year=year_filter,
                filter_full_text_only=full_text_only
            )
        
        # добавление ответа ассистента в историю
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "sources": result['sources'] if result['has_sources'] else []
        })
        
        # перезагрузка страницы для отображения истории
        st.rerun()
    
    # футер
    st.markdown("---")
    st.markdown("""
    <div class="footer">
         Alzheimer RAG Assistant | поиск по научным статьям из PubMed | 
         LLM: {}</div>
    """.format("включена" if st.session_state.use_llm else "выключена"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()