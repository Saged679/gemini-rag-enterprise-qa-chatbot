import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

os.makedirs("data/raw_docs", exist_ok=True)
os.makedirs("vectorstore/chroma", exist_ok=True)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True
)

embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# QA prompt — chat_history is injected as real messages so the LLM can answer
# both document questions AND conversational meta-questions (e.g. "what did I ask first?")
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.

Rules:
- If the question is about the current conversation (e.g. "what was my first question?", \
"what did I ask earlier?", "summarize our chat"), answer using the conversation history — \
do NOT say you don't know.
- If the retrieved context below is relevant to the question, use it to answer.
- For any other question, answer from your own general knowledge.

Retrieved Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

qa_chain = qa_prompt | llm


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


st.set_page_config(page_title="📘 Enterprise Policy Q&A Chatbot", layout="wide")
st.title("📘 Enterprise Policy Chatbot")
st.caption("Ask anything about HR, IT Security, or Code of Conduct")

# Session state initialization
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

if "upload_vectorstore" not in st.session_state:
    st.session_state.upload_vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Default vectorstore from persisted chroma (hardcoded PDFs)
default_vectorstore = Chroma(
    persist_directory="vectorstore/chroma",
    embedding_function=embed_model
)

# Upload UI
st.sidebar.header("📄 Upload New PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        raw_docs = []
        for file in new_files:
            file_path = os.path.join("data/raw_docs", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyMuPDFLoader(file_path)
            raw_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(raw_docs)

        if st.session_state.upload_vectorstore is None:
            st.session_state.upload_vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embed_model
            )
        else:
            st.session_state.upload_vectorstore.add_documents(chunks)

        st.session_state.processed_files.update(f.name for f in new_files)
        st.sidebar.success(f"✅ {len(new_files)} file(s) indexed.")

if st.session_state.upload_vectorstore is not None:
    st.sidebar.info(f"📂 Answering from {len(st.session_state.processed_files)} uploaded file(s)")
    if st.sidebar.button("🗑️ Clear uploads, use default docs"):
        st.session_state.upload_vectorstore = None
        st.session_state.processed_files = set()
        st.rerun()

# Use uploaded vectorstore exclusively when available, otherwise fall back to default
active_vectorstore = (
    st.session_state.upload_vectorstore
    if st.session_state.upload_vectorstore is not None
    else default_vectorstore
)
retriever = active_vectorstore.as_retriever(search_kwargs={"k": 4})

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
        chat_history = st.session_state.chat_memory.load_memory_variables({})["chat_history"]
        source_docs = retriever.invoke(query)
        context = format_docs(source_docs)

        answer = qa_chain.invoke({
            "question": query,
            "context": context,
            "chat_history": chat_history,
        }).content

        st.session_state.chat_memory.save_context(
            {"input": query},
            {"output": answer},
        )

    st.chat_message("assistant", avatar="🤖").markdown(answer)

    with st.expander("📚 Sources"):
        if source_docs:
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Source {i+1}**")
                if hasattr(doc, "page_content") and isinstance(doc.page_content, str):
                    st.code(doc.page_content.strip()[:1000] + "...", language="markdown")
                else:
                    st.code("Content not available or not in expected format.", language="markdown")
        else:
            st.markdown("No sources found for this query.")
