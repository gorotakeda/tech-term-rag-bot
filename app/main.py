from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from app.rag_system import ReactDocRAG

# 環境変数の読み込み
load_dotenv()

app = FastAPI()

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAGシステムのインスタンス化
rag_system = ReactDocRAG()


class Query(BaseModel):
    question: str


@app.get("/")
async def root():
    return {"message": "React Documentation RAG API"}


@app.post("/query")
async def query_docs(query: Query):
    try:
        answer = rag_system.query(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
