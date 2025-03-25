import pandas as pd
import re
import os

# Пути к файлам
csv_files = [
    "/kaggle/input/kazakh-krgz-russian-english-nlp/KG_news_256k.csv",
    "/kaggle/input/kazakh-krgz-russian-english-nlp/kazakhBooks.csv",
    "/kaggle/input/kazakh-krgz-russian-english-nlp/kazakhNews.csv"
]
txt_file = "/kaggle/input/kazakh-krgz-russian-english-nlp/kk_wiki_articles.txt"
output_path = "merged_dataset.txt"

# Компилируем регулярные выражения один раз
url_pattern = re.compile(r"http\S+")
clean_pattern = re.compile(r"[^a-zA-Zа-яА-ЯёЁәғқңөұүһӘҒҚҢӨҰҮҺ0-9\s\.\,\!\?\;\:\-\(\)\"\'«»]")
whitespace_pattern = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Очищает текст, удаляя ссылки, лишние символы и пробелы."""
    text = url_pattern.sub("", text)
    text = clean_pattern.sub("", text)
    text = whitespace_pattern.sub(" ", text)
    return text.strip()

def process_csv_file(csv_file: str, chunksize: int = 10000) -> list:
    """Читает CSV файл чанками и возвращает список очищенных текстов из колонки 'text'."""
    texts = []
    for chunk in pd.read_csv(csv_file, usecols=["text"], dtype=str, encoding="utf-8", chunksize=chunksize):
        # Заполняем пропуски пустыми строками и очищаем текст
        cleaned = chunk["text"].fillna("").apply(clean_text).tolist()
        texts.extend(cleaned)
    return texts

def process_txt_file(txt_file: str) -> list:
    """Построчно читает txt файл и возвращает список очищенных строк."""
    texts = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                texts.append(clean_text(stripped_line))
    return texts

def main():
    all_texts = []

    # Обработка CSV-файлов с использованием чанков
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"Обрабатываю {csv_file}...")
            texts = process_csv_file(csv_file)
            all_texts.extend(texts)
        else:
            print(f"Файл {csv_file} не найден, пропускаем.")

    # Обработка txt-файла
    if os.path.exists(txt_file):
        print(f"Обрабатываю {txt_file}...")
        all_texts.extend(process_txt_file(txt_file))
    else:
        print(f"Файл {txt_file} не найден.")

    # Запись в единый файл
    with open(output_path, "w", encoding="utf-8") as f_out:
        # Добавляем перевод строки к каждой строке
        f_out.writelines(line + "\n" for line in all_texts if line)
    
    print(f"Объединённый файл сохранён как {output_path}. Всего строк: {len(all_texts)}")

if __name__ == "__main__":
    main()
