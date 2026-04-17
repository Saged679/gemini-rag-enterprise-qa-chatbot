import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Ensure directories exist
os.makedirs("data/raw_docs", exist_ok=True)
os.makedirs("vectorstore/chroma", exist_ok=True)

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True
)

embed_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Streamlit UI Setup
st.set_page_config(page_title="📘 Enterprise Policy Q&A Chatbot", layout="wide")
st.title("📘 Enterprise Policy Chatbot")
st.caption("Ask anything about HR, IT Security, or Code of Conduct")

# Chat Memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Load VectorStore
vectorstore = Chroma(
    persist_directory="vectorstore/chroma",
    embedding_function=embed_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Define RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.chat_memory,
    return_source_documents=True,
    output_key="answer"
)

# Upload UI
st.sidebar.header("📄 Upload New PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    raw_docs = []
    for file in uploaded_files:
        file_path = os.path.join("data/raw_docs", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyMuPDFLoader(file_path)
        raw_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    vectorstore.add_documents(chunks)

    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) added and indexed.")

# Chat interface
st.divider()

messages = st.session_state.chat_memory.chat_memory.messages
for msg in messages:
    if hasattr(msg, "type") and msg.type == "human":
        st.chat_message("user", avatar="🧑").markdown(msg.content)
    elif hasattr(msg, "type") and msg.type == "ai":
        st.chat_message("assistant", avatar="🤖").markdown(msg.content)

query = st.chat_input("Type your question here...")

if query:
    st.chat_message("user", avatar="🧑").markdown(query)

    with st.spinner("🤔 Thinking..."):
        result = qa_chain.invoke({"question": query})
        answer = result["answer"]
        sources = result.get("source_documents", [])

    st.chat_message("assistant", avatar="🤖").markdown(answer)

    with st.expander("📚 Sources"):
        if sources:
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}**")
                if hasattr(doc, "page_content") and isinstance(doc.page_content, str):
                    st.code(doc.page_content.strip()[:1000] + "...", language="markdown")
                else:
                    st.code("Content not available or not in expected format.", language="markdown")
        else:
            st.markdown("No sources found for this query.")