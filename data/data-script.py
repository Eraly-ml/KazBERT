import pandas as pd
import re
import os

# ПУТИ К ФАЙЛАМ
csv_files = [
    "/kaggle/input/kazakh-krgz-russian-english-nlp/KG_news_256k.csv",
    "/kaggle/input/kazakh-krgz-russian-english-nlp/kazakhBooks.csv",
    "/kaggle/input/kazakh-krgz-russian-english-nlp/kazakhNews.csv"
]
txt_file = "/kaggle/input/kazakh-krgz-russian-english-nlp/kk_wiki_articles.txt"

# ФУНКЦИЯ ОЧИСТКИ ТЕКСТА
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)  # Удаляем ссылки
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁәғқңөұүһӘҒҚҢӨҰҮҺ0-9\s\.\,\!\?\;\:\-\(\)\"\'«»]", "", text)
    text = re.sub(r"\s+", " ", text)  # Удаляем лишние пробелы
    return text.strip()

def main():
    all_texts = []

    # 1) Обрабатываем CSV-файлы
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, usecols=["text"], dtype=str, encoding="utf-8")
            cleaned_texts = df["text"].fillna("").apply(clean_text).tolist()
            all_texts.extend(cleaned_texts)

    # 2) Обрабатываем txt-файл
    if os.path.exists(txt_file):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            all_texts.extend([clean_text(line) for line in lines if line.strip()])

    # 3) Сохраняем в единый txt-файл
    output_path = "merged_dataset.txt"
    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in all_texts:
            f_out.write(line + "\n")

    print(f"Объединённый файл сохранён как {output_path}. Всего строк: {len(all_texts)}")

if __name__ == "__main__":
    main()
