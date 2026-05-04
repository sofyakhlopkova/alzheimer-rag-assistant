# src/llm/ollama_client.py
import logging
import requests
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config.settings import config


def query_llm(prompt: str, 
                 model: str = config.LLM_MODEL,
                 temperature: float = config.TEMPERATURE,
                 max_tokens: int = config.MAX_TOKENS, num_ctx: int = config.NUM_CONTEXT) -> Optional[str]:
    
    """отправляет запрос к локальной Ollama"""

    url = config.URL_GENERATE
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        answer = result.get("response", "")
        logging.info(f"llm ответила, длина: {len(answer)} символов")
        return answer
        
    except requests.exceptions.ConnectionError:
        logging.error("llm не запущен!")
        return None
    except Exception as e:
        logging.error(f"ошибка при запросе к llm: {e}")
        return None


def check_llm_available() -> bool:
    """проверяет доступность llm"""
    try:
        response = requests.get(config.URL_TAGS, timeout=5)
        return response.status_code == 200
    except:
        return False
