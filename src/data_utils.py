import requests
import pandas as pd
from pathlib import Path
import re


def tokenize_function(texts, tokenizer, max_length=128):
    return tokenizer(
        texts.tolist() if hasattr(texts, 'tolist') else texts, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length,
        return_tensors="pt"
    )

def clean_text(text, min_lenght):
    """ Очистка текста """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    
    # замена упоминаний @username и ссылок
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # оставляем только буквы, цифры и пробелы
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # удаления избытков пробелов и очистка краев
    text = re.sub(r"\s+", " ", text).strip()

    if len(text.split()) < min_lenght:
        return ""
    
    return text

def load_raw_data(url: str, save_path: Path):
    """ Загрузка сырых данных и разбивка на твиты """

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"Файл уже существует: {save_path}")
        return 

    try:
        response = requests.get(url)
        response.raise_for_status() 

        content = response.text

        tweets = re.split(r'\n(?=@)', content)

        df = pd.DataFrame({'tweet_text': tweets})
        df.to_csv(save_path, index=False, encoding='utf-8', quoting=1)
        
        print(f"Обработано твитов: {len(df)}")
        print('Файл успешно сохранен:', save_path)
        
    except Exception as e:
        print(f"Ошибка при обработке: {e}")