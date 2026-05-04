from sentence_transformers import SentenceTransformer
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config.settings import config

def load_embedding_model(model_name: str = config.EMBEDDING_MODEL):

    model = SentenceTransformer(model_name)
    logging.info(f"модель {model_name} загружена, размер эмбеддингов: {model.get_sentence_embedding_dimension()}")
    return model