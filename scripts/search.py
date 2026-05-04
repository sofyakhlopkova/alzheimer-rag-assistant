import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import config
from src.database.chroma_client import init_chroma_db
from src.database.collection import create_or_get_collection
from src.embedding.model_manager import load_embedding_model
from src.database.vector_store import search_similar, format_results

def main():
    client = init_chroma_db(config.DB_DIRECTORY)
    collection = create_or_get_collection(client, config.COLLECTION_NAME)
    model = load_embedding_model(config.EMBEDDING_MODEL)
    
    query = input("вопрос: ")
    results = search_similar(query, collection, model, n_results=5)
    df = format_results(results)
    print(df.to_string())

if __name__ == "__main__":
    main()