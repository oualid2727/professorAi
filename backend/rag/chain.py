from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain.chains import RetrievalQA


def build_retriever(chroma_host: str = "chromadb", chroma_port: int = 8000):
    vectorstore = Chroma(
        collection_name="ai_professor",
        persist_directory="/chroma",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        client_settings={
            "chroma_server_host": chroma_host,
            "chroma_server_http_port": chroma_port,
        },
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def build_chain():
    llm = ChatOllama(
        model="llama3",
        temperature=0.1,
        system=(
            "You are a specialized Professor. Answer only based on the "
            "provided course context. If the answer isn't in the context, "
            "politely say you haven't covered that topic yet."
        ),
    )
    retriever = build_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
    return qa_chain


def answer_question(query: str) -> dict:
    chain = build_chain()
    result = chain({"query": query})
    return {
        "answer": result.get("result", ""),
        "sources": [doc.metadata for doc in result.get("source_documents", [])],
    }

