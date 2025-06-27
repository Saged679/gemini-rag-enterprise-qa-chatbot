import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Step 0: Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Step 1: Load and chunk all PDFs
raw_dir = "data/raw_docs"
all_chunks = []

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir(raw_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(raw_dir, filename)
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

print(f"✅ Total chunks loaded and split: {len(all_chunks)}")

# Step 2: Initialize Gemini Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Step 3: Create and persist Chroma Vector Store
persist_dir = "vectorstore/chroma"

vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=persist_dir
)

vectorstore.persist()
print(f"✅ Vectorstore persisted at: {persist_dir}")
