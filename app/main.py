from fastapi import FastAPI, APIRouter
import uvicorn
from app.services.service_ner_gherman import GhermanNerService
from app.services.service_russian_spell_corrector import RussianSpellCorrectorService
from contextlib import asynccontextmanager
from fastapi import Request


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Загрузка моделей...")
    app.state.gherman_ner = GhermanNerService()
    app.state.ru_sp_corrector = RussianSpellCorrectorService()
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
    cor_text = model_cor.predict(text)
    ner_text = model_ner.predict(cor_text)
    return {"input_text": text, "ner_text": ner_text}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
