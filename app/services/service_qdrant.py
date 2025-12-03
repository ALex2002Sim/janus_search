import re
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer


class QdrantDB:
    def __init__(
        self,
        host="localhost",
        port=6333,
        collection_name="addresses",
        data: List[str] = None,
        batch_size: int = 1000,
        timeout: int = 120,
    ):
        self.model_name = "ai-forever/sbert_large_nlu_ru"
        self.model = SentenceTransformer(self.model_name)

        self.vector_size = self.model.get_sentence_embedding_dimension()

        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.collection_name = collection_name
        self.batch_size = batch_size

        collections = [c.name for c in self.client.get_collections().collections]

        if collection_name in collections:
            print(
                f"Коллекция '{collection_name}' уже существует. Используем её и не загружаем данные."
            )
            self._collection_exists = True
        else:
            print(
                f"Коллекция '{collection_name}' не найдена. Создаём новую и загружаем данные."
            )
            self.create_collection()
            self._collection_exists = False
            if data:
                self.upload_items(data)

    def create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "address_vector": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            },
        )
        print(f"Коллекция '{self.collection_name}' создана.")

    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        self.create_collection()


    @staticmethod
    def clean_address(addr: str) -> str:
        """
        Минимальная нормализация — можно улучшать позже.
        Удаляем почтовые индексы, лишние пробелы.
        """
        addr = re.sub(r"\b\d{6}\b", "", addr)
        addr = re.sub(r"\s+", " ", addr)
        return addr.strip()


    def upload_items(self, items: List[str]):
        if self._collection_exists:
            print(
                f"Коллекция '{self.collection_name}' уже существует. Данные не будут загружены."
            )
            return

        total = len(items)
        print(f"Загружаем {total} адресов...")

        for start in range(0, total, self.batch_size):
            batch = items[start:start + self.batch_size]
            vectors = self.model.encode(batch, show_progress_bar=False)

            points = []
            for i, (addr, vec) in enumerate(zip(batch, vectors), start=start):
                clean = self.clean_address(addr)
                points.append(
                    {
                        "id": i,
                        "vector": {"address_vector": vec.tolist()},
                        "payload": {
                            "address": clean,
                            "raw_address": addr,
                        },
                    }
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Пакет {start}-{start + len(batch)} загружен ({len(batch)} записей)")

        print("Готово!")

    def search_address(self, query: str, limit=5):
        clean = self.clean_address(query)
        vec = self.model.encode(clean).tolist()

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            using="address_vector",
            limit=limit,
        )
        return result.points



if __name__ == "__main__":
    import json

    with open("address_cleaned.json", "r", encoding="utf-8") as f:
        loaded_items = json.load(f)

    db = QdrantDB(collection_name="addresses_cleaned_sbert", data=loaded_items, timeout=300)
