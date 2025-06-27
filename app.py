import os
import streamlit as st
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, convert_system_message_to_human=True)
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Streamlit UI Setup
st.set_page_config(page_title="📘 Enterprise Policy Q&A Chatbot", layout="wide")
st.title("📘 Enterprise Policy Chatbot")
st.caption("Ask anything about HR, IT Security, or Code of Conduct")

# Chat Memory (Session-State Preserved)
if "chat_memory" not in st.session_state:
    # Set the 'output_key' for the ConversationBufferMemory to match the 'answer' key
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' # Explicitly set output_key to 'answer'
    )

# Load VectorStore (Chroma)
vectorstore = Chroma(persist_directory="vectorstore/chroma", embedding_function=embed_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Define the RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.chat_memory,
    return_source_documents=True,
    output_key='answer' # Explicitly set output_key to 'answer' here as well
)

# Upload UI (multi-file)
st.sidebar.header("📄 Upload New PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    raw_docs = []
    for file in uploaded_files:
        with open(f"data/raw_docs/{file.name}", "wb") as f:
            f.write(file.getbuffer())
        loader = PyMuPDFLoader(f"data/raw_docs/{file.name}")
        raw_docs.extend(loader.load())

    # Chunk new files and add to vectorstore
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    vectorstore.add_documents(chunks)
    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) added and indexed.")

# --- Chat Interface ---
st.divider()
for msg in st.session_state.get("chat_history", []):
    # Ensure to check if msg.type exists before accessing, though for HumanMessage and AIMessage it should.
    # Also, ensure 'content' is the correct attribute for displaying the message.
    if hasattr(msg, 'type') and msg.type == "human":
        st.chat_message("user", avatar="🧑").markdown(msg.content)
    elif hasattr(msg, 'type') and msg.type == "ai": # Langchain's AI messages usually have type 'ai'
        st.chat_message("assistant", avatar="🤖").markdown(msg.content)
    else: # Fallback for unexpected message types or initial setup issues
        st.chat_message("assistant", avatar="🤖").markdown(str(msg)) # display raw message if type is unknown

query = st.chat_input("Type your question here...")

if query:
    st.chat_message("user", avatar="🧑").markdown(query)

    with st.spinner("🤔 Thinking..."):
        result = qa_chain.invoke({"question": query})
        answer = result["answer"]
        sources = result.get("source_documents", [])

    st.chat_message("assistant", avatar="🤖").markdown(answer)

    # Show highlighted sources if available
    with st.expander("📚 Sources"):
        if sources:
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}**")
                # Ensure that doc.page_content exists and is a string
                if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                    st.code(doc.page_content.strip()[:1000] + "...", language="markdown")
                else:
                    st.code("Content not available or not in expected format.", language="markdown")
        else:
            st.markdown("No sources found for this query.")