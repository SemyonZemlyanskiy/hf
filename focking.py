from sentence_transformers import SentenceTransformer

# Загрузка модели (при первом запуске она скачивается и кэшируется)
model = SentenceTransformer('all-MiniLM-L6-v2-main-focking')

sentences = ["Привет, мир!", "Это тест."]
embeddings = model.encode(sentences)

# Пример: вывод векторов
for i, emb in enumerate(embeddings):
    print(f"Sentence {i}: {emb}")