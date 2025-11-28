from fastapi import FastAPI, APIRouter
import uvicorn
from app.services.service_ner_gherman import GhermanNer
from contextlib import asynccontextmanager
from fastapi import Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Загрузка моделей...")
    app.state.gherman_ner = GhermanNer(model_name = "Gherman/bert-base-NER-Russian")
    print("Все модели готовы!")    
    yield 

app = FastAPI(lifespan=lifespan)

@app.get("/gherman_ner")
def extract_address(request: Request, text: str):
    model = request.app.state.gherman_ner
    return model.predict(text)

@app.get("/")
def classify_text():
    return {"message": "Hi"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)