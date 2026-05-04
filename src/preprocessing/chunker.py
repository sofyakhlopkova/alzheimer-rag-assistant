import pandas as pd
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config.settings import config


def split_into_chunks(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list:
    """
    разбиение текста на чанки с перекрытием 
    """
    if pd.isna(text) or text == '':
        return []
    
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    step = chunk_size - overlap
    
    for i in range(0, len(words), step): # start, stop, step
        chunk = ' '.join(words[i:i + chunk_size]) #  срез списка
        if chunk:
            chunks.append(chunk)
    
    return chunks


def create_chunks(df: pd.DataFrame, 
                  use_full_text: bool = True,
                  chunk_size: int = config.CHUNK_SIZE,
                  overlap: int = config.CHUNK_OVERLAP) -> pd.DataFrame:

    logging.info(f"разбиение документов на чанки")
    
    text_column = 'full_text' if use_full_text else 'search_text'
    source_type = "полный текст" if use_full_text else "заголовок+абстракт"
    
    all_chunks = []
    docs_without_text = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="создание чанков"):
        text = row[text_column]
        
        if pd.isna(text) or text == '':
            docs_without_text += 1
            continue
        
        chunks = split_into_chunks(text, chunk_size, overlap)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_data = {
                'chunk_id': f"{row['doc_id']}_chunk_{chunk_idx}",
                'doc_id': row['doc_id'],
                'pmid': row['pmid'],
                'title': row['title'],
                'year': row['year'],
                'has_full_text': row['has_full_text'],
                'chunk_text': chunk_text,
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks),
                'source_type': source_type,
                'is_full_text': use_full_text
            }
            all_chunks.append(chunk_data)
    
    chunks_df = pd.DataFrame(all_chunks)
    
    logging.info(f"создано {len(chunks_df)} чанков из {len(df) - docs_without_text} документов")
    logging.info(f"документов без текста: {docs_without_text}")
    
    return chunks_df