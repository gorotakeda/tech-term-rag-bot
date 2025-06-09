import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
from typing import List


class ReactDocRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
        )
        self.vectorstore = None
        self.qa_chain = None

    def fetch_react_docs(self) -> List[str]:
        """Reactの公式ドキュメントを取得"""
        # ここでは例としてReactのメインページを使用
        url = "https://react.dev/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # テキストの抽出（実際の実装ではより詳細なスクレイピングが必要）
        texts = []
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            if element.text.strip():
                texts.append(element.text.strip())
        return texts

    def initialize(self):
        """RAGシステムの初期化"""
        # ドキュメントの取得と分割
        texts = self.fetch_react_docs()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_text("\n".join(texts))

        # ベクトルストアの作成
        self.vectorstore = Chroma.from_texts(
            chunks, self.embeddings, persist_directory="./chroma_db"
        )

        # QAチェーンの設定
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                model_name="gpt-4.1-nano",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            ),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
        )

    def query(self, question: str) -> str:
        """質問に対する回答を生成"""
        if not self.qa_chain:
            self.initialize()
        return self.qa_chain.run(question)
