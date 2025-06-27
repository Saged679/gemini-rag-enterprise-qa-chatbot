import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
# Ensure the API key is set for LangChain/Google GenAI
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="Enterprise Policy Chatbot API")

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WARNING: Use specific frontend domains in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema for the API endpoint
class Question(BaseModel):
    query: str
    session_id: str = "default" # Allows for multiple chat sessions

# Initialize Embeddings and Retriever globally to avoid re-loading on each request
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# The Chroma class itself is used the same way, only the import path changes.
vectorstore = Chroma(persist_directory="vectorstore/chroma", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# In-memory dictionary to store chat memory for each session.
# In a real-world production app, consider using a persistent store like Redis or a database
# for session management.
session_memories = {}

@app.post("/ask/")
async def ask_question(data: Question):
    """
    Handles incoming chat questions, processes them using the RAG chain,
    and returns the answer along with source documents.
    """
    session_id = data.session_id

    # Initialize a new ConversationBufferMemory for the session if it doesn't exist
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            # IMPORTANT: Explicitly set output_key to 'answer' to resolve ValueError
            output_key="answer"
        )

    memory = session_memories[session_id]
    
    # Initialize LLM within the function, although it could also be global if no
    # dynamic parameters (like temperature) are expected per request.
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

    # Define the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        # IMPORTANT: Explicitly set output_key to 'answer' for the chain
        output_key="answer"
    )

    try:
        # Invoke the QA chain with the user's question
        result = qa_chain.invoke({"question": data.query})
        
        # Extract the answer and source documents
        answer = result["answer"]
        
        # Safely get source documents and their metadata
        sources = []
        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                # Ensure doc.metadata exists and has a 'source' key
                source_info = doc.metadata.get("source", "unknown")
                sources.append(source_info)

        return {
            "session_id": session_id,
            "question": data.query,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        # Basic error handling for API responses
        return {
            "session_id": session_id,
            "question": data.query,
            "answer": f"An error occurred: {str(e)}",
            "sources": []
        }
