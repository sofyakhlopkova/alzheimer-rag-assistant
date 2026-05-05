import pandas as pd
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.database.chroma_client import init_chroma_db
from src.database.collection import create_or_get_collection
from src.preprocessing.chunker import create_chunks
from src.embedding.model_manager import load_embedding_model
from src.preprocessing.prepare import prepare_documents

from src.config.settings import config

def add_to_vector_db(chunks_df: pd.DataFrame, 
                     collection,
                     embedding_model,
                     batch_size: int = 100):
    """
    добавление чанков в векторную БД
    """
    logging.info("добавление документов в ChromaDB")
    
    # проверяем, есть ли уже данные
    if collection.count() > 0:
        logging.warning(f"коллекция уже содержит {collection.count()} документов")
        response = input("очистить коллекцию? (y/n): ").lower()
        if response == 'y':
            collection.delete(ids=collection.get()['ids'])
            logging.info("коллекция очищена")
        else:
            logging.info("добавляем новые документы к существующим")
    
    # добавление батчами
    total_batches = (len(chunks_df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(chunks_df), batch_size), total=total_batches, desc="добавление в БД"):
        batch = chunks_df.iloc[i:i+batch_size]
        
        # создание эмбеддингов
        texts = batch['chunk_text'].tolist()
        embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()
        
        # ID и метаданные
        ids = batch['chunk_id'].tolist()
        metadatas = []
        for _, row in batch.iterrows():
            metadata = {
                'pmid': str(row['pmid']),
                'title': str(row['title'])[:] if row['title'] else '',
                'year': int(row['year']) if pd.notna(row['year']) else 0,
                'has_full_text': bool(row['has_full_text']),
                'chunk_index': int(row['chunk_index']),
                'total_chunks': int(row['total_chunks']),
                'source_type': str(row['source_type'])
            }
            metadatas.append(metadata)
        
        # добавление в коллекцию
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    logging.info(f"добавлено {collection.count()} документов в коллекцию")


def create_vector_store(df: pd.DataFrame, 
                        persist_directory: str = config.DB_DIRECTORY,
                        collection_name: str = config.COLLECTION_NAME,
                        use_full_text: bool = True,
                        chunk_size: int = 300,
                        chunk_overlap: int = 50,
                        model_name: str = config.EMBEDDING_MODEL,
                        path: str = config.PATH_CHUNKS_CSV,):
    """
    основная функция для создания векторного хранилища
    """
    # инициализация
    client = init_chroma_db(persist_directory)
    embedding_model = load_embedding_model(model_name)
    collection = create_or_get_collection(client, collection_name)
    
    # подготовка документов
    df = prepare_documents(df)
    
    # создание чанков
    chunks_df = create_chunks(
        df, 
        use_full_text=use_full_text,
        chunk_size=chunk_size,
        overlap=chunk_overlap
    )
    
    # добавление в векторную БД
    add_to_vector_db(chunks_df, collection, embedding_model)
    
    # сохранение информации о чанках
    chunks_df.to_csv(path, index=False)
    logging.info(f"чанки сохранены в {path}")
    
    return collection, chunks_df


def search_similar(query: str, 
                   collection, 
                   embedding_model,
                   n_results: int = 5,
                   filter_by_year: tuple = None,
                   filter_full_text_only: bool = False):
    """
    поиск похожих документов с фильтрацией
    """
    where_filter = None
    
    conditions = []
    
    if filter_full_text_only:
        conditions.append({'has_full_text': True})
    
    if filter_by_year and len(filter_by_year) == 2:
        min_year, max_year = filter_by_year
        conditions.append({'year': {'$gte': min_year}})
        conditions.append({'year': {'$lte': max_year}})
    
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {'$and': conditions}
    
    # эмбеддинг запроса
    query_embedding = embedding_model.encode(query).tolist()
    
    # поиск
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter  # теперь может быть None, словарь или $and
    )
    
    return results


def format_results(results: dict) -> pd.DataFrame:
    """
    форматирование результатов в DataFrame
    """
    formatted = []
    
    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i]
        
        # Пропускаем если метаданные None
        if metadata is None:
            continue
        
        similarity = 1 - results['distances'][0][i]
        
        result = {
            'pmid': metadata.get('pmid', ''),
            'title': metadata.get('title', '')[:],
            'year': metadata.get('year', 0),
            'similarity': similarity,
            'source_type': metadata.get('source_type', ''),
            'text_preview': results['documents'][0][i][:200] + '...' if results['documents'][0][i] else ''
        }
        formatted.append(result)
    
    return pd.DataFrame(formatted)