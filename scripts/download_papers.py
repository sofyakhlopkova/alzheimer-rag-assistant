import requests
import time
import logging
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET  # для парсинга

# настройка логирования 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

from src.config.settings import config
from functools import wraps

def rate_limit(calls_per_second=3):
    """декоратор для ограничения частоты запросов"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 1.0 / calls_per_second - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(3)
def search_pubmed(query: str, retmax: int = 50, email: str = config.EMAIL) -> List[str]:
    """
    поиск PMIDs в PubMed по запросу
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance"
    }
    
    headers = {
        "User-Agent": f"AlzheimerRAG (mailto:{email})"
    }
    
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        
        logging.info(f"PubMed: найдено {len(pmids)} статей")
        return pmids
        
    except Exception as e:
        logging.error(f"ошибка поиска в PubMed: {e}")
        return []

@rate_limit(3)
def fetch_abstracts(pmids: List[str], email: str = config.EMAIL) -> List[Dict]:
    """
    получение абстрактов по PMIDs
    """
    if not pmids:
        return []
    
    params = {
        "db": "pubmed",
        "id": ",".join(pmids[:50]),
        "retmode": "xml"
    }
    
    headers = {
        "User-Agent": f"AlzheimerRAG (mailto:{email})"
    }
    
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        # парсинг XML
        articles = []
        root = ET.fromstring(response.text)
        
        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.findtext(".//PMID", "")
                title = article.findtext(".//ArticleTitle", "")
                
                # собираем абстракт 
                abstract_parts = []
                for abstract_elem in article.findall(".//AbstractText"):
                    if abstract_elem.text:
                        abstract_parts.append(abstract_elem.text)
                abstract = " ".join(abstract_parts)
                
                journal = article.findtext(".//Journal/Title", "")
                year = article.findtext(".//PubDate/Year", "")
                
                if pmid and title:
                    articles.append({
                        "source": "PubMed",
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "year": year,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
                    
            except Exception as e:
                logging.error(f"ошибка парсинга статьи: {e}")
                continue
        
        logging.info(f"получено {len(articles)} абстрактов")
        return articles
        
    except Exception as e:
        logging.error(f"ошибка получения абстрактов: {e}")
        return []

@rate_limit(3)
def get_pmc_id_from_pmid(pmid: str, email: str = config.EMAIL) -> Optional[str]:
    """
    получение PMC ID по PMID (чтобы узнать, есть ли полный текст)
    """
    params = {
        "db": "pubmed",
        "linkname": "pubmed_pmc",
        "from_uid": pmid,
        "retmode": "json"
    }
    
    headers = {
        "User-Agent": f"AlzheimerRAG (mailto:{email})"
    }
    
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        linksets = data.get("linksets", [])
        if linksets and "linksetdbs" in linksets[0]:
            for link in linksets[0]["linksetdbs"]:
                if link.get("linkname") == "pubmed_pmc":
                    pmc_ids = link.get("links", [])
                    if pmc_ids:
                        return f"PMC{pmc_ids[0]}"
        return None
        
    except Exception as e:
        logging.error(f"ошибка получения PMC ID для {pmid}: {e}")
        return None

@rate_limit(3)
def fetch_full_text_from_pmc(pmc_id: str, email: str = config.EMAIL) -> Dict[str, str]:
    """
    получение полного текста из PMC (если доступен)
    """
    if not pmc_id:
        return {}
    
    pmc_id = pmc_id.replace("PMC", "")
    
    params = {
        "db": "pmc",
        "id": pmc_id,
        "retmode": "xml"
    }
    
    headers = {
        "User-Agent": f"AlzheimerRAG (mailto:{email})"
    }
    
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        # парсим секции
        sections = {"introduction": "", "conclusion": ""}
        root = ET.fromstring(response.text)
        
        # ищем body статьи
        body = root.find(".//body")
        if body is not None:
            current_text = []
            current_section = ""
            
            for elem in body.iter():
                if elem.tag == "sec":
                    # сохраняем предыдущую секцию
                    if current_section and current_text:
                        if "intro" in current_section or "background" in current_section:
                            sections["introduction"] += " ".join(current_text)
                        elif "conclu" in current_section or "discuss" in current_section:
                            sections["conclusion"] += " ".join(current_text)
                    
                    # начинаем новую секцию
                    title = elem.findtext(".//title", "").lower()
                    current_section = title
                    current_text = []
                    
                elif elem.tag == "p" and elem.text:
                    current_text.append(elem.text)
            
            # последняя секция
            if current_section and current_text:
                if "conclu" in current_section or "discuss" in current_section:
                    sections["conclusion"] += " ".join(current_text)
        
        return sections
        
    except Exception as e:
        logging.error(f"ошибка получения полного текста для {pmc_id}: {e}")
        return {}


def collect_alzheimer_papers(
    queries: List[str] = None,
    max_papers: int = 50,
    email: str = "your@email.com"
) -> List[Dict]:
    """
    главная функция для сбора статей
    """
    if queries is None:
        queries = [
            "Alzheimer's disease targets",
            "Alzheimer therapeutic targets", 
            "Alzheimer drug targets"
        ]
    
    all_articles = []
    seen_pmids = set()
    
    logging.info("сбор данных")
    
    for query in queries:
        if len(all_articles) >= max_papers:
            break
            
        logging.info(f"\nзапрос: {query}")
        
        # ищем PMIDs
        pmids = search_pubmed(query, retmax=30, email=email)
        
        if pmids:
            # фильтруем уже собранные
            new_pmids = [p for p in pmids if p not in seen_pmids]
            
            if new_pmids:
                # получаем абстракты
                articles = fetch_abstracts(new_pmids[:15], email=email)
                
                for article in articles:
                    if article["pmid"] not in seen_pmids and len(all_articles) < max_papers:
                        # проверяем, есть ли полный текст
                        pmc_id = get_pmc_id_from_pmid(article["pmid"], email)
                        
                        if pmc_id:
                            # добавляем введение и заключение
                            full_text = fetch_full_text_from_pmc(pmc_id, email)
                            article["introduction"] = full_text.get("introduction", "")
                            article["conclusion"] = full_text.get("conclusion", "")
                            article["has_full_text"] = True
                            logging.info(f"получен полный текст для {article['pmid']}")
                        else:
                            article["introduction"] = ""
                            article["conclusion"] = ""
                            article["has_full_text"] = False
                            logging.info(f"только абстракт для {article['pmid']}")
                        
                        article["query"] = query
                        all_articles.append(article)
                        seen_pmids.add(article["pmid"])
        
        time.sleep(1)
        logging.info(f"прогресс: {len(all_articles)}/{max_papers}")
    
   
    logging.info(f"собрано {len(all_articles)} статей")
    logging.info(f"   - с полным текстом: {sum(1 for a in all_articles if a.get('has_full_text'))}")
    
    return all_articles

def save_articles_to_csv(articles: List[Dict], filename: str = config.PATH_RAW_ARTICLES_CSV):
    """
    сохранение статей в CSV
    """
    import pandas as pd
    import os
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False, encoding='utf-8')
    
    logging.info(f"данные сохранены в {filename}")
    
    # статистика
    print("\nстатистика:")
    print(f"всего статей: {len(df)}")
    print(f"с абстрактами: {df['abstract'].str.len().gt(50).sum()}")
    print(f"с полным текстом: {df['has_full_text'].sum() if 'has_full_text' in df else 0}")
    if 'year' in df.columns:
        print(f"годы: {df['year'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    
    papers = collect_alzheimer_papers(
        max_papers=100,
        email=config.EMAIL
    )
    
    if papers:
        save_articles_to_csv(papers)
    else:
        logging.error("не удалось собрать статьи")