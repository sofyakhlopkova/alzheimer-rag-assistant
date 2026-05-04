import logging               
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config.settings import config

def create_or_get_collection(client, collection_name: str = config.COLLECTION_NAME):
    """
    создание или получение существующей коллекции
    """
    try:
        collection = client.get_collection(name=collection_name)
        logging.info(f"коллекция '{collection_name}' загружена, всего документов: {collection.count()}")
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # косинусное расстояние (по умолчанию)
        )
        logging.info(f"создана новая коллекция '{collection_name}'")
    
    return collection