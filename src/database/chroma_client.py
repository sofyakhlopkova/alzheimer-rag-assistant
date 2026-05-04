import chromadb                    # основная библиотека ChromaDB
from chromadb.config import Settings # настройки клиента
import logging                     # для логирования
from pathlib import Path           # для работы с путями файловой системы
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config.settings import config

def init_chroma_db(persist_directory: str = config.DB_DIRECTORY):

    Path(persist_directory).mkdir(exist_ok=True) # создает папку для хранения БД
    
    client = chromadb.PersistentClient(
        path=persist_directory,
    )
    
    logging.info(f"ChromaDB инициализирована в {persist_directory}")
    return client