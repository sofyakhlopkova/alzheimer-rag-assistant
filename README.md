# Alzheimer RAG Assistant

RAG-система для поиска научных статей по болезни Альцгеймера.

## Возможности

- Поиск по научным статьям через векторную базу ChromaDB
- Интеграция с Ollama (qwen2.5:1.5b)
- Streamlit интерфейс
- Фильтрация по годам и наличию полного текста
- История диалога

## Установка

```bash
# Клонирование
git clone https://github.com/sofyakhlopkova/alzheimer-rag-assistant.git
cd alzheimer-rag-assistant

# Виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Зависимости
pip install -r requirements.txt

# Ollama
ollama pull qwen2.5:1.5b
