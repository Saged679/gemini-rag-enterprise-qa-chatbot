import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="Enterprise Policy Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str
    session_id: str = "default"

# Initialize Embeddings and default vectorstore globally
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
default_vectorstore = Chroma(persist_directory="vectorstore/chroma", embedding_function=embedding)

# Per-session state
session_memories = {}
session_vectorstores = {}  # Holds uploaded-PDF vectorstores keyed by session_id


@app.post("/upload/")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    session_id: str = Form(default="default")
):
    """
    Upload one or more PDFs and index them for a specific session.
    Questions sent with the same session_id will be answered from these files only.
    """
    os.makedirs("data/raw_docs", exist_ok=True)
    raw_docs = []

    for file in files:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            continue
        file_path = os.path.join("data/raw_docs", filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        loader = PyMuPDFLoader(file_path)
        raw_docs.extend(loader.load())

    if not raw_docs:
        return {"status": "error", "message": "No valid PDF files found."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    if session_id in session_vectorstores:
        session_vectorstores[session_id].add_documents(chunks)
    else:
        session_vectorstores[session_id] = Chroma.from_documents(
            documents=chunks,
            embedding=embedding
        )

    return {
        "status": "success",
        "session_id": session_id,
        "files_indexed": len(files),
        "chunks_created": len(chunks)
    }


@app.post("/ask/")
async def ask_question(data: Question):
    """
    Handles incoming chat questions using the RAG chain.
    Uses session-specific uploaded vectorstore if available, otherwise falls back to default.
    """
    session_id = data.session_id

    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    memory = session_memories[session_id]

    # Use session-specific vectorstore (from uploads) if available, else default
    active_vectorstore = session_vectorstores.get(session_id, default_vectorstore)
    retriever = active_vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    try:
        result = qa_chain.invoke({"question": data.query})
        answer = result["answer"]

        sources = []
        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                sources.append(doc.metadata.get("source", "unknown"))

        return {
            "session_id": session_id,
            "question": data.query,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "question": data.query,
            "answer": f"An error occurred: {str(e)}",
            "sources": []
        }
