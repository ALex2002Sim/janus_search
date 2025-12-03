from fastapi import FastAPI, APIRouter
import uvicorn
from app.services.service_ner_gherman import GhermanNerService
from app.services.service_russian_spell_corrector import RussianSpellCorrectorService
from app.services.service_qdrant import QdrantDB
from contextlib import asynccontextmanager
from fastapi import Request


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Загрузка моделей...")
    app.state.gherman_ner = GhermanNerService()
    app.state.ru_sp_corrector = RussianSpellCorrectorService()
    app.state.qdrant_client = QdrantDB(collection_name="addresses_cleaned_sbert")
    print("Все модели готовы!")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/gherman_ner")
def extract_address(request: Request, text: str):
    model = request.app.state.gherman_ner
    return {"input_text": text, "ner_text": model.predict(text)}


@app.get("/ru_sp_corrector")
def extract_address(request: Request, text: str):
    model = request.app.state.ru_sp_corrector
    return {"input_text": text, "cor_text": model.predict(text)}


@app.get("/adress_extract")
def classify_text(request: Request, text: str):
    model_cor = request.app.state.ru_sp_corrector
    model_ner = request.app.state.gherman_ner
    qdrant_conn = request.app.state.qdrant_client

    print(f"Ввод пользователя: {text}\n")

    cor_text = model_cor.predict(text)
    print(f"Ввод пользователя после исправления ошибок: {cor_text}\n")

    ner_text = model_ner.predict(cor_text)
    print(f"Вывод NER: {ner_text}\n")

    if not ner_text:
        ner_text = cor_text

    print(f"Результаты из бд")
    results = qdrant_conn.search_address(query=ner_text)
    for p in results:
        print(f"{p.payload}\n{p.score}\n")

    return {"input_text": text, "processed_text": results[0].payload["raw_address"]}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)
