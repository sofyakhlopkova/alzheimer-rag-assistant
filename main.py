import pandas as pd
import logging
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config.settings import config

from src.preprocessing.prepare import prepare_dataset
from src.database.vector_store import create_vector_store

def main():
    """
    основная функция
    """
    # загружаем исходные данные
    df = pd.read_csv(config.PATH_RAW_ARTICLES_CSV)
    
    # предобрабатываем
    processed_df = prepare_dataset(df)
    
    # сохраняем
    processed_df.to_csv(config.PATH_PREPARED_PAPERS_CSV, index=False)
    
    # сохраняем отдельно для RAG 
    rag_df = processed_df[['pmid', 'title', 'cleaned_abstract', 'cleaned_text', 
                           'year', 'has_full_text']]
    rag_df.to_csv(config.PATH_RAG_READY_PAPERS_CSV, index=False)
    
    print(f"данные подготовлены, сохранено {len(rag_df)} статей для RAG")


    try:
        df = pd.read_csv(config.PATH_RAG_READY_PAPERS_CSV)
        logging.info(f"загружено {len(df)} статей")
        print("\nколонки в данных:", list(df.columns))
    except FileNotFoundError:
        logging.error(f"файл {config.PATH_RAG_READY_PAPERS_CSV} не найден")
        return
    
    # создание векторного хранилища
    collection, chunks_df = create_vector_store(
        df=df,
        persist_directory=config.DB_DIRECTORY,
        collection_name=config.COLLECTION_NAME,
        use_full_text=config.USE_FULL_TEXT,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        model_name=config.EMBEDDING_MODEL
    )
    
    # сохранение информации
    storage_info = {
        'collection_name': collection.name,
        'total_chunks': collection.count(),
        'total_articles': len(df),
        'articles_with_full_text': int(df['has_full_text'].sum()),
        'chunk_size': config.CHUNK_SIZE,
        'chunk_overlap': config.CHUNK_OVERLAP,
        'use_full_text': config.USE_FULL_TEXT,
        'embedding_model': config.EMBEDDING_MODEL,
        'year_range': [int(df['year'].min()), int(df['year'].max())]
    }
    
    with open(config.PATH_STORAGE_INFO, 'w', encoding='utf-8') as f:
        json.dump(storage_info, f, indent=2)

if __name__ == "__main__":
    main()