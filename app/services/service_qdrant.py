import re
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, QueryRequest
from sentence_transformers import SentenceTransformer


class AddressDB:
    def __init__(
        self,
        host="localhost",
        port=6333,
        collection_name="addresses",
        vector_size=384,
        data: List[dict] = None,
        batch_size: int = 1000,
        timeout: int = 120,
    ):
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = batch_size

        collections = [c.name for c in self.client.get_collections().collections]

        if collection_name in collections:
            print(
                f"Коллекция '{collection_name}' уже существует. Используем её и не загружаем данные."
            )
            self._collection_exists = True
        else:
            print(
                f"Коллекция '{collection_name}' не найдена. Создаём новую и загружаем данные, если переданы."
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
                    size=self.vector_size, distance=Distance.COSINE
                )
            },
        )
        print(f"Коллекция '{self.collection_name}' создана.")

    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        self.create_collection()

    @staticmethod
    def clean_address(addr: str) -> str:
        return re.sub(r"\b\d{6}\b", "", addr).strip()

    def upload_items(self, items: List[dict]):
        if self._collection_exists:
            print(
                f"Коллекция '{self.collection_name}' уже существует. Данные не будут загружены."
            )
            return

        total = len(items)
        for start in range(0, total, self.batch_size):
            batch = items[start : start + self.batch_size]
            points = []
            for i, item in enumerate(batch, start=start):
                addr = self.clean_address(item)
                vec = self.model.encode(addr).tolist()
                points.append(
                    {
                        "id": i,
                        "vector": {"address_vector": vec},
                        "payload": {"address": addr, "raw_address": item},
                    }
                )
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Пакет {start}-{start + len(batch)} загружен ({len(batch)} записей)")

    def search_address(self, query: str, limit=5):
        q = self.clean_address(query)
        vec = self.model.encode(q).tolist()
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            using="address_vector",
            limit=limit,
        )
        return result.points


if __name__ == "__main__":
    import json

    with open("address.json", "r", encoding="utf-8") as f:
        loaded_items = json.load(f)
    db = AddressDB(collection_name="addresses", data=loaded_items, timeout=300)
