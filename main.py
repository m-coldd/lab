#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа для парсинга новостей с BBC News и создания пайплайна предобработки данных для ML
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class NewsParser:
    """Класс для парсинга новостей с BBC News"""
    
    def __init__(self):
        self.base_url = "https://www.bbc.com"
        self.news_url = "https://www.bbc.com/news"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_news_by_count(self, count=10):
        """
        Получить указанное количество новостей
        
        Args:
            count (int): Количество новостей для парсинга
            
        Returns:
            pd.DataFrame: DataFrame с новостями
        """
        print(f"Парсинг {count} новостей с BBC News...")
        
        news_data = []
        page = 1
        
        while len(news_data) < count:
            try:
                # Получаем страницу с новостями
                url = f"{self.news_url}?page={page}"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Находим все ссылки на новости
                news_links = soup.find_all('a', href=True)
                
                for link in news_links:
                    if len(news_data) >= count:
                        break
                        
                    href = link.get('href')
                    if href and '/news/' in href and href.startswith('/'):
                        full_url = self.base_url + href
                        
                        # Получаем заголовок
                        title_element = link.find(['h1', 'h2', 'h3', 'span'])
                        if title_element:
                            title = title_element.get_text(strip=True)
                            
                            if title and len(title) > 10:  # Фильтруем короткие заголовки
                                # Получаем полный текст новости
                                article_text = self._get_article_text(full_url)
                                
                                news_data.append({
                                    'title': title,
                                    'url': full_url,
                                    'text': article_text,
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'word_count': len(article_text.split()) if article_text else 0
                                })
                                
                                print(f"Получено новостей: {len(news_data)}/{count}")
                                time.sleep(0.5)  # Пауза между запросами
                
                page += 1
                
            except Exception as e:
                print(f"Ошибка при парсинге страницы {page}: {e}")
                break
        
        return pd.DataFrame(news_data[:count])
    
    def get_news_by_date(self, start_date, count=50):
        """
        Получить новости начиная с указанной даты
        
        Args:
            start_date (str): Дата в формате 'YYYY-MM-DD'
            count (int): Максимальное количество новостей
            
        Returns:
            pd.DataFrame: DataFrame с новостями
        """
        print(f"Парсинг новостей с {start_date}...")
        
        news_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # BBC News имеет архивы по датам
        for i in range(30):  # Проверяем последние 30 дней
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                # Формируем URL для конкретной даты
                url = f"{self.news_url}/{date_str}"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Находим новости на странице
                    articles = soup.find_all('article') or soup.find_all('div', class_=re.compile(r'news|article'))
                    
                    for article in articles[:10]:  # Ограничиваем количество с одной страницы
                        if len(news_data) >= count:
                            break
                            
                        title_elem = article.find(['h1', 'h2', 'h3'])
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            
                            if title and len(title) > 10:
                                link_elem = article.find('a', href=True)
                                if link_elem:
                                    href = link_elem.get('href')
                                    if href.startswith('/'):
                                        full_url = self.base_url + href
                                    else:
                                        full_url = href
                                    
                                    article_text = self._get_article_text(full_url)
                                    
                                    news_data.append({
                                        'title': title,
                                        'url': full_url,
                                        'text': article_text,
                                        'date': date_str,
                                        'word_count': len(article_text.split()) if article_text else 0
                                    })
                
                current_date += timedelta(days=1)
                time.sleep(1)  # Пауза между запросами
                
            except Exception as e:
                print(f"Ошибка при парсинге даты {date_str}: {e}")
                continue
        
        return pd.DataFrame(news_data[:count])
    
    def _get_article_text(self, url):
        """
        Получить полный текст статьи
        
        Args:
            url (str): URL статьи
            
        Returns:
            str: Текст статьи
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Ищем основной контент статьи
            content_selectors = [
                'div[data-component="text-block"]',
                '.story-body',
                '.article-body',
                'div[property="articleBody"]',
                '.content'
            ]
            
            article_text = ""
            for selector in content_selectors:
                content_divs = soup.select(selector)
                if content_divs:
                    for div in content_divs:
                        paragraphs = div.find_all('p')
                        for p in paragraphs:
                            text = p.get_text(strip=True)
                            if text and len(text) > 20:  # Фильтруем короткие абзацы
                                article_text += text + " "
                    break
            
            return article_text.strip()
            
        except Exception as e:
            print(f"Ошибка при получении текста статьи {url}: {e}")
            return ""


class DataPreprocessingPipeline:
    """Класс для создания пайплайна предобработки данных"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        
    def visualize_data(self, df):
        """Создание визуализаций для анализа данных"""
        print("Создание визуализаций...")
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Распределение длины заголовков
        df['title_length'] = df['title'].str.len()
        axes[0, 0].hist(df['title_length'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Распределение длины заголовков')
        axes[0, 0].set_xlabel('Длина заголовка')
        axes[0, 0].set_ylabel('Частота')
        
        # 2. Распределение количества слов в тексте
        axes[0, 1].hist(df['word_count'], bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Распределение количества слов в тексте')
        axes[0, 1].set_xlabel('Количество слов')
        axes[0, 1].set_ylabel('Частота')
        
        # 3. Корреляция между признаками
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
            axes[1, 0].set_title('Корреляционная матрица')
        else:
            axes[1, 0].text(0.5, 0.5, 'Недостаточно числовых признаков', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Корреляционная матрица')
        
        # 4. Количество новостей по датам
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            daily_counts = df.groupby(df['date'].dt.date).size()
            axes[1, 1].plot(daily_counts.index, daily_counts.values, marker='o')
            axes[1, 1].set_title('Количество новостей по датам')
            axes[1, 1].set_xlabel('Дата')
            axes[1, 1].set_ylabel('Количество новостей')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('news_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def convert_to_numeric(self, df):
        """Преобразование данных к числовым типам"""
        print("Преобразование данных к числовым типам...")
        
        df_processed = df.copy()
        
        # Создаем числовые признаки из текстовых данных
        df_processed['title_length'] = df_processed['title'].str.len()
        df_processed['text_length'] = df_processed['text'].str.len()
        df_processed['word_count'] = df_processed['text'].str.split().str.len()
        df_processed['sentence_count'] = df_processed['text'].str.count(r'[.!?]+')
        
        # Создаем признаки на основе даты
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'])
            df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
            df_processed['month'] = df_processed['date'].dt.month
            df_processed['year'] = df_processed['date'].dt.year
        
        # Создаем бинарные признаки
        df_processed['has_url'] = df_processed['url'].notna().astype(int)
        df_processed['has_text'] = (df_processed['text'].str.len() > 0).astype(int)
        
        return df_processed
    
    def handle_missing_values(self, df):
        """Обработка пропущенных значений"""
        print("Обработка пропущенных значений...")
        
        df_processed = df.copy()
        
        # Заполняем пропуски в текстовых полях
        df_processed['title'] = df_processed['title'].fillna('Unknown Title')
        df_processed['text'] = df_processed['text'].fillna('No text available')
        df_processed['url'] = df_processed['url'].fillna('No URL')
        
        # Заполняем пропуски в числовых полях
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
            df_processed[numeric_columns].mean()
        )
        
        return df_processed
    
    def normalize_data(self, df):
        """Нормализация и стандартизация данных"""
        print("Нормализация и стандартизация данных...")
        
        df_processed = df.copy()
        
        # Выбираем числовые колонки для нормализации
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['day_of_week', 'month', 'year']]
        
        if len(numeric_columns) > 0:
            # Стандартизация (z-score normalization)
            df_processed[numeric_columns] = self.scaler.fit_transform(df_processed[numeric_columns])
            
            # Создаем версию с Min-Max нормализацией для сравнения
            minmax_scaler = MinMaxScaler()
            df_processed[[f'{col}_minmax' for col in numeric_columns]] = minmax_scaler.fit_transform(
                df_processed[numeric_columns]
            )
        
        return df_processed
    
    def create_ml_features(self, df):
        """Создание дополнительных признаков для ML"""
        print("Создание дополнительных признаков для ML...")
        
        df_processed = df.copy()
        
        # TF-IDF признаки из заголовков (упрощенная версия)
        from collections import Counter
        
        # Собираем все слова из заголовков
        all_words = []
        for title in df_processed['title']:
            words = re.findall(r'\b\w+\b', title.lower())
            all_words.extend(words)
        
        # Находим наиболее частые слова
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(20)]
        
        # Создаем бинарные признаки для частых слов
        for word in common_words:
            df_processed[f'contains_{word}'] = df_processed['title'].str.lower().str.contains(word).astype(int)
        
        # Создаем признаки на основе длины текста
        df_processed['is_long_article'] = (df_processed['word_count'] > df_processed['word_count'].median()).astype(int)
        df_processed['is_short_article'] = (df_processed['word_count'] < df_processed['word_count'].quantile(0.25)).astype(int)
        
        return df_processed
    
    def run_full_pipeline(self, df):
        """Запуск полного пайплайна предобработки"""
        print("Запуск полного пайплайна предобработки данных...")
        
        # 1. Визуализация исходных данных
        df = self.visualize_data(df)
        
        # 2. Преобразование к числовым типам
        df = self.convert_to_numeric(df)
        
        # 3. Обработка пропусков
        df = self.handle_missing_values(df)
        
        # 4. Нормализация
        df = self.normalize_data(df)
        
        # 5. Создание ML признаков
        df = self.create_ml_features(df)
        
        print(f"Пайплайн завершен. Итоговая форма данных: {df.shape}")
        return df


def main():
    """Основная функция"""
    print("=== Программа парсинга новостей и создания пайплайна ML ===\n")
    
    # Создаем парсер новостей
    parser = NewsParser()
    
    # Выбор режима работы
    print("Выберите режим работы:")
    print("1. Парсинг по количеству новостей")
    print("2. Парсинг по дате")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == "1":
        count = int(input("Введите количество новостей для парсинга: "))
        news_df = parser.get_news_by_count(count)
    elif choice == "2":
        date = input("Введите дату начала парсинга (YYYY-MM-DD): ")
        count = int(input("Введите максимальное количество новостей: "))
        news_df = parser.get_news_by_date(date, count)
    else:
        print("Неверный выбор. Используется режим по умолчанию (10 новостей).")
        news_df = parser.get_news_by_count(10)
    
    # Сохраняем исходные данные
    news_df.to_csv('raw_news_data.csv', index=False, encoding='utf-8')
    print(f"\nИсходные данные сохранены в 'raw_news_data.csv'")
    print(f"Получено новостей: {len(news_df)}")
    
    # Показываем пример данных
    print("\nПример данных:")
    print(news_df.head())
    
    # Создаем пайплайн предобработки
    pipeline = DataPreprocessingPipeline()
    
    # Запускаем полный пайплайн
    processed_df = pipeline.run_full_pipeline(news_df)
    
    # Сохраняем обработанные данные
    processed_df.to_csv('processed_news_data.csv', index=False, encoding='utf-8')
    print(f"\nОбработанные данные сохранены в 'processed_news_data.csv'")
    
    # Показываем статистику
    print("\nСтатистика обработанных данных:")
    print(f"Количество строк: {processed_df.shape[0]}")
    print(f"Количество признаков: {processed_df.shape[1]}")
    print(f"Пропущенные значения: {processed_df.isnull().sum().sum()}")
    
    # Показываем типы данных
    print("\nТипы данных:")
    print(processed_df.dtypes.value_counts())
    
    print("\nПрограмма завершена успешно!")


if __name__ == "__main__":
    main()
