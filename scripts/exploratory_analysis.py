import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

def load_data(file_path: str) -> pd.DataFrame:
    """загрузка данных"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Загружено {len(df)} статей из {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"Файл {file_path} не найден")
        raise

def calculate_text_stats(df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
    """
    расчет базовых статистик по текстам
    """
    logging.info("расчет базовых статистик")
    
    def get_stats(text):
        if pd.isna(text) or text == '':
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'unique_words': 0
            }
        
        words = text.split()
        sentences = nltk.sent_tokenize(text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'unique_words': len(set(words))
        }
    
    # применяем статистики к каждой статье
    stats_df = df[text_column].apply(get_stats).apply(pd.Series)
    
    # добавляем производные метрики
    stats_df['words_per_sentence'] = stats_df['word_count'] / stats_df['sentence_count']
    stats_df['lexical_diversity'] = stats_df['unique_words'] / stats_df['word_count']
    
    # заменяем inf и NaN
    stats_df = stats_df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # объединяем с исходным датафреймом
    result_df = pd.concat([df, stats_df], axis=1)
    
    logging.info("Статистики рассчитаны")
    return result_df
    
def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """
    визуализация распределений метрик
    """
    logging.info("построение распределений")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. распределение количества слов
    axes[0, 0].hist(df['word_count'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Распределение количества слов в статье', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Количество слов')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].axvline(df['word_count'].median(), color='red', linestyle='--', 
                       label=f'Медиана: {df["word_count"].median():.0f}')
    axes[0, 0].legend()
    
    # 2. распределение количества предложений
    axes[0, 1].hist(df['sentence_count'].dropna(), bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Распределение количества предложений', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Количество предложений')
    axes[0, 1].set_ylabel('Частота')
    
    # 3. распределение длины предложений
    axes[0, 2].hist(df['words_per_sentence'].dropna(), bins=50, color='salmon', edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Распределение средней длины предложения (слов)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Слов в предложении')
    axes[0, 2].set_ylabel('Частота')
    
    # 4. лексическое разнообразие
    axes[1, 0].hist(df['lexical_diversity'].dropna(), bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Лексическое разнообразие\n(уникальные слова / всего слов)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Коэффициент разнообразия')
    axes[1, 0].set_ylabel('Частота')
    
    # 5. boxplot количества слов
    axes[1, 1].boxplot(df['word_count'].dropna())
    axes[1, 1].set_title('Boxplot количества слов', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Количество слов')
    
    # 6. распределение по годам 
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().sort_index()
        axes[1, 2].plot(year_counts.index, year_counts.values, marker='o', linestyle='-', color='darkblue')
        axes[1, 2].set_title('Динамика публикаций по годам', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Год')
        axes[1, 2].set_ylabel('Количество статей')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Нет данных по годам', ha='center', va='center')
        axes[1, 2].set_title('Годы не указаны', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_vocabulary(df: pd.DataFrame, text_column: str, output_dir: Path, 
                       n_top: int = 50, language: str = 'english'):
    """
    анализ словарного состава
    """
    logging.info("Анализ словарного состава...")
    
    # объединяем все тексты
    all_text = ' '.join(df[text_column].dropna())
    words = nltk.word_tokenize(all_text.lower())
    
    # фильтруем слова (только буквенные)
    words = [word for word in words if word.isalpha() and len(word) > 2]
    
    # получаем стоп-слова
    stop_words = set(stopwords.words(language))
    
    # слова без стоп-слов
    content_words = [word for word in words if word not in stop_words]
    
    # частотные распределения
    all_words_freq = Counter(words)
    content_words_freq = Counter(content_words)
    
    # визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # топ слова 
    top_words, top_counts = zip(*all_words_freq.most_common(n_top))
    axes[0].barh(range(n_top), top_counts, color='skyblue')
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(top_words)
    axes[0].set_title(f'Топ-{n_top} самых частых слов\n(включая стоп-слова)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Частота')
    axes[0].invert_yaxis()
    
    # топ слова (только значимые)
    top_words_c, top_counts_c = zip(*content_words_freq.most_common(n_top))
    axes[1].barh(range(n_top), top_counts_c, color='lightcoral')
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(top_words_c)
    axes[1].set_title(f'Топ-{n_top} значимых слов\n(без стоп-слов)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Частота')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_words.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # анализ биграмм и триграмм
    analyze_ngrams(words, stop_words, output_dir, n=2, n_top=20)
    analyze_ngrams(words, stop_words, output_dir, n=3, n_top=20)
    
    return {
        'top_words_all': all_words_freq.most_common(10),
        'top_words_content': content_words_freq.most_common(10)
    }

def analyze_ngrams(words: list, stop_words: set, output_dir: Path, 
                   n: int = 2, n_top: int = 20):
    """
    анализ n-грамм
    """
    logging.info(f"анализ {n}-грамм")
    
    # создаем n-граммы
    n_grams = list(ngrams(words, n))
    
    # фильтруем n-граммы, содержащие стоп-слова
    filtered_ngrams = [ng for ng in n_grams if not any(w in stop_words for w in ng)]
    
    ngram_freq = Counter([' '.join(ng) for ng in filtered_ngrams])
    
    # визуализация
    plt.figure(figsize=(12, 8))
    top_ngrams, top_counts = zip(*ngram_freq.most_common(n_top))
    
    plt.barh(range(n_top), top_counts, color='teal')
    plt.yticks(range(n_top), top_ngrams)
    plt.title(f'Топ-{n_top} самых частых {n}-грамм', fontsize=14, fontweight='bold')
    plt.xlabel('Частота')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(output_dir / f'top_{n}grams.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_wordclouds(df: pd.DataFrame, text_column: str, output_dir: Path, 
                      language: str = 'english'):
    """
    создание облаков слов
    """
    logging.info("создание облаков слов")
    
    all_text = ' '.join(df[text_column].dropna())
    
    # облако для всех слов
    wordcloud_all = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=200,
        colormap='viridis'
    ).generate(all_text)
    
    # облако только для значимых слов
    words = nltk.word_tokenize(all_text.lower())
    words = [word for word in words if word.isalpha() and len(word) > 2]
    stop_words = set(stopwords.words(language))
    content_words = [word for word in words if word not in stop_words]
    content_text = ' '.join(content_words)
    
    wordcloud_content = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=200,
        colormap='plasma'
    ).generate(content_text)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(wordcloud_all, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Облако слов (все слова)', fontsize=14, fontweight='bold')
    
    axes[1].imshow(wordcloud_content, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Облако слов (только значимые)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_by_year(df: pd.DataFrame, output_dir: Path):
    """
    анализ динамики текстов по годам
    """
    if 'year' not in df.columns:
        logging.warning("колонка 'year' не найдена.")
        return
    
    logging.info("анализ динамики по годам")
    
    # группируем по годам
    yearly_stats = df.groupby('year').agg({
        'word_count': ['mean', 'count'],
        'lexical_diversity': 'mean'
    }).round(2)
    
    yearly_stats.columns = ['avg_words', 'num_articles', 'avg_lexical_diversity']
    yearly_stats = yearly_stats.reset_index()
    
    # визуализация
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # количество статей по годам
    axes[0, 0].bar(yearly_stats['year'], yearly_stats['num_articles'], color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Количество статей по годам', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Год')
    axes[0, 0].set_ylabel('Количество статей')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # средняя длина статьи по годам
    axes[0, 1].plot(yearly_stats['year'], yearly_stats['avg_words'], marker='o', linestyle='-', color='red')
    axes[0, 1].set_title('Средняя длина статьи по годам', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Год')
    axes[0, 1].set_ylabel('Среднее количество слов')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # лексическое разнообразие по годам
    axes[1, 0].plot(yearly_stats['year'], yearly_stats['avg_lexical_diversity'], marker='s', linestyle='-', color='green')
    axes[1, 0].set_title('Лексическое разнообразие по годам', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Год')
    axes[1, 0].set_ylabel('Среднее лексическое разнообразие')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # тепловая карта корреляций
    corr_matrix = df[['word_count', 'sentence_count', 'words_per_sentence', 'lexical_diversity']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Корреляция метрик', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yearly_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return yearly_stats

def detect_outliers(df: pd.DataFrame, threshold: float = 1.5):
    """
    обнаружение выбросов в текстах
    """
    logging.info("поиск выбросов")
    
    # используем IQR метод для количества слов
    Q1 = df['word_count'].quantile(0.25)
    Q3 = df['word_count'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df['word_count'] < lower_bound) | (df['word_count'] > upper_bound)]
    
    logging.info(f"найдено {len(outliers)} статей-выбросов по длине")
    
    if len(outliers) > 0:
        print(f"Статьи короче {lower_bound:.0f} слов или длиннее {upper_bound:.0f} слов:")
        print(f"Всего выбросов: {len(outliers)} ({len(outliers)/len(df)*100:.1f}% от всех статей)")

        print("\nПримеры статей-выбросов:")
        for idx, row in outliers.head(5).iterrows():
            print(f"  - PMID: {row.get('pmid', 'N/A')}, Слов: {row['word_count']}, Заголовок: {str(row.get('title', ''))[:80]}...")
    
    return outliers, lower_bound, upper_bound

def generate_report(df: pd.DataFrame, vocab_results: dict, yearly_stats, 
                    outliers_info: tuple, output_dir: Path):
    """
    генерация отчета по анализу
    """
    logging.info("генерация отчета")
    
    outliers, lower_bound, upper_bound = outliers_info
    
    report = []
    
    # Общая информация
    report.append(f"\n1. Общая информация")
    report.append(f"   - Всего статей: {len(df)}")
    if 'year' in df.columns:
        report.append(f"   - Диапазон годов: {df['year'].min()} - {df['year'].max()}")
        report.append(f"   - Статей с полным текстом: {df['has_full_text'].sum() if 'has_full_text' in df.columns else 'N/A'}")
    
    # Статистики текстов
    report.append(f"\n2. Статистики текстов")
    stats = df[['word_count', 'sentence_count', 'words_per_sentence', 'lexical_diversity']].describe()
    report.append(stats.to_string())
    
    # Медианные значения
    report.append(f"\n   Медианные значения:")
    report.append(f"   - Медианное количество слов: {df['word_count'].median():.0f}")
    report.append(f"   - Медианное количество предложений: {df['sentence_count'].median():.0f}")
    report.append(f"   - Медианная длина предложения: {df['words_per_sentence'].median():.1f} слов")
    report.append(f"   - Медианное лексическое разнообразие: {df['lexical_diversity'].median():.3f}")
    
    # Топ слова
    report.append(f"\n3. Топ-10 значимых слов")
    for word, count in vocab_results['top_words_content'][:10]:
        report.append(f"   - {word}: {count}")
    
    # Выбросы
    report.append(f"\n4. Выбросы")
    report.append(f"   - Количество статей-выбросов: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    report.append(f"   - Нижняя граница: {lower_bound:.0f} слов")
    report.append(f"   - Верхняя граница: {upper_bound:.0f} слов")
    
    # Годовая статистика
    if yearly_stats is not None:
        report.append(f"\n5. Динамика по годам")
        report.append(f"   - Год с наибольшим количеством статей: {yearly_stats.loc[yearly_stats['num_articles'].idxmax(), 'year']} "
                     f"({yearly_stats['num_articles'].max()} статей)")
        report.append(f"   - Год с самыми длинными статьями: {yearly_stats.loc[yearly_stats['avg_words'].idxmax(), 'year']} "
                     f"({yearly_stats['avg_words'].max():.0f} слов в среднем)")
        report.append(f"   - Год с самым высоким лексическим разнообразием: {yearly_stats.loc[yearly_stats['avg_lexical_diversity'].idxmax(), 'year']} "
                     f"({yearly_stats['avg_lexical_diversity'].max():.3f})")
    
    # Сохраняем отчет
    report_text = '\n'.join(report)
    with open(output_dir / 'eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text

def main():
    """
    основная функция для EDA
    """
    # создаем директорию для результатов
    output_dir = Path('results/eda_results')
    output_dir.mkdir(exist_ok=True)
    
    # загружаем подготовленные данные
    input_file = 'data/processed/alzheimer_papers_cleaned.csv'
    
    try:
        df = load_data(input_file)
    except FileNotFoundError:
        print("Файл не найден. Сначала запустите prepare_data.py")
        return
    
    # 1. Базовые статистики
    df = calculate_text_stats(df, text_column='cleaned_text')
    
    # 2. Визуализация распределений
    plot_distributions(df, output_dir)
    
    # 3. Анализ словарного состава
    vocab_results = analyze_vocabulary(df, 'cleaned_text', output_dir, 
                                       n_top=30, language='english')
    
    # 4. Облака слов
    create_wordclouds(df, 'cleaned_text', output_dir, language='english')
    
    # 5. Анализ по годам (если есть данные)
    yearly_stats = analyze_by_year(df, output_dir)
    
    # 6. Поиск выбросов
    outliers_info = detect_outliers(df)
    
    # 7. Генерация отчета
    generate_report(df, vocab_results, yearly_stats, outliers_info, output_dir)
    
    # Сохраняем обогащенный датафрейм
    enriched_output = 'data/processed/alzheimer_papers_with_metrics.csv'
    df.to_csv(enriched_output, index=False)
    logging.info(f"Обогащенный датафрейм сохранен в {enriched_output}")
    
    print(f"\n EDA завершен. Результаты сохранены в директории '{output_dir}/'")

if __name__ == "__main__":
    main()