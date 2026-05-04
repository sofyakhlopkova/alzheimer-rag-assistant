# src/config/settings.py - упрощенная версия
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional

import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# переменные в os.environ вручную
if ENV_FILE.exists():
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

class Settings(BaseSettings):
    """конфигурация"""
    
    # пути
    DB_DIRECTORY: str = Field('chroma_db', env='DB_DIRECTORY')
    PATH_RAW_ARTICLES_CSV: str = Field('data/raw/alzheimer_papers.csv', env='PATH_RAW_ARTICLES_CSV')
    PATH_PREPARED_PAPERS_CSV: str = Field('data/processed/prepared_papers.csv', env='PATH_PREPARED_PAPERS_CSV')
    PATH_RAG_READY_PAPERS_CSV: str = Field('data/processed/rag_ready_papers.csv', env='PATH_RAG_READY_PAPERS_CSV')
    PATH_STORAGE_INFO: str = Field('chroma_db/storage_info.json', env='PATH_STORAGE_INFO')
    PATH_CHUNKS_CSV: str = Field('data/processed/document_chunks.csv', env='PATH_CHUNKS_CSV')
    
    # параметры
    CHUNK_SIZE: int = Field(300, env='CHUNK_SIZE')
    CHUNK_OVERLAP: int = Field(50, env='CHUNK_OVERLAP')
    USE_FULL_TEXT: bool = Field(True, env='USE_FULL_TEXT')
    
    # модели
    EMBEDDING_MODEL: str = Field('intfloat/multilingual-e5-small', env='EMBEDDING_MODEL')
    LLM_MODEL: str = Field('qwen2.5:1.5b', env='LLM_MODEL')
    COLLECTION_NAME: str = Field('alzheimer_papers', env='COLLECTION_NAME')
    
    # LLM параметры
    TEMPERATURE: float = Field(0.3, env='TEMPERATURE')
    MAX_TOKENS: int = Field(512, env='MAX_TOKENS')
    MAX_CHUNKS: int = Field(5, env='MAX_CHUNKS')
    MAX_TEXT_LENGTH: int = Field(500, env='MAX_TEXT_LENGTH')
    NUM_CONTEXT: int = Field(2048, env='NUM_CONTEXT')
    
    # Ollama URLs
    URL_GENERATE: str = Field('http://localhost:11434/api/generate', env='URL_GENERATE')
    URL_TAGS: str = Field('http://localhost:11434/api/tags', env='URL_TAGS')
    
    # API
    EMAIL: Optional[str] = Field(None, env='EMAIL')
    PUBMED_API_KEY: str = Field('', env='PUBMED_API_KEY')
    
    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = 'ignore'  # игнорируем лишние поля

settings = Settings()
config = settings  # для обратной совместимости