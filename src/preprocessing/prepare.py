import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import Any

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    подготовка документов 
    """
    # текст для индексации (title + abstract)
    df['search_text'] = df['title'].fillna('') + ' ' + df['cleaned_abstract'].fillna('')
    
    # для полного текста 
    df['full_text'] = df['cleaned_text'].fillna('')
    
    # уникальный ID для каждого документа из pmid
    df['doc_id'] = df['pmid'].astype(str)
    
    df['year'] = df['year'].fillna(0).astype(int)
    
    # флаг наличия полного текста
    df['has_full_text'] = df['has_full_text'].fillna(False).astype(bool)
    
    logging.info(f"документы подготовлены, всего: {len(df)}")
    logging.info(f"  - с полным текстом: {df['has_full_text'].sum()}")
    logging.info(f"  - диапазон годов: {df['year'].min()} - {df['year'].max()}")
    
    return df
  

def convert_to_string(value: Any) -> str:
    """
    преобразование любого значения в строку
    """
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return str(int(value)) if value == int(value) else str(value)
    if isinstance(value, str):
        return value
    return str(value)


def clean_text(text: Any) -> str:
    """очистка текста"""
    text = convert_to_string(text)
    
    if not text:
        return ""
    
    # HTML теги если есть
    text = re.sub(r'<[^>]+>', '', text)
    
    # спецсимволы, только буквы и базовая пунктуация
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)]', '', text)
    
    # лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def preprocess_for_rag(text: Any) -> str:
    """
    предобработка для RAG 
    """
    text = convert_to_string(text)
    
    if not text:
        return ""
    
    # разбиваем на предложения
    sentences = sent_tokenize(text)
    
    # очищаем каждое предложение
    cleaned_sentences = [clean_text(sent) for sent in sentences]
    
    # фильтруем слишком короткие предложения
    cleaned_sentences = [s for s in cleaned_sentences if len(s.split()) > 5]
    
    return ' '.join(cleaned_sentences)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    основная функция подготовки данных
    """
    logging.info("предобработка данных")
    
    # создаем копию
    processed_df = df.copy()
    
    # объединяем текст (абстракт + введение + заключение)
    processed_df['full_text'] = (
        processed_df['abstract'].fillna('') + ' ' + 
        processed_df['introduction'].fillna('') + ' ' + 
        processed_df['conclusion'].fillna('')
    )
    
    # очищаем тексты
    processed_df['cleaned_text'] = processed_df['full_text'].apply(preprocess_for_rag)
    processed_df['cleaned_abstract'] = processed_df['abstract'].apply(preprocess_for_rag)
    
    # добавляем мета-информацию для RAG
    processed_df['chunk_size'] = processed_df['cleaned_text'].str.len()
    
    # удаляем статьи без текста
    processed_df = processed_df[processed_df['cleaned_text'].str.len() > 100]
    
    logging.info(f"предобработка завершена, осталось статей: {len(processed_df)}")
    
    return processed_df