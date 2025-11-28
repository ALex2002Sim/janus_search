from fastapi import FastAPI, APIRouter
import uvicorn

app = FastAPI()


@app.get("/")
def classify_text():
    return {"message": "Hi"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)