import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Model + Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

# Load Vector Store
persist_dir = "vectorstore/chroma"
# The Chroma class itself is used the same way, only the import path changes.
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Enable memory
# IMPORTANT: Explicitly set output_key to 'answer' to resolve ValueError
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer" # This tells the memory which key to use for the AI's response
)

# Conversational RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    # IMPORTANT: Explicitly set output_key for the chain as well for consistency
    output_key="answer"
)

# --- Terminal Chat Loop ---
print("✅ RAG Q&A System with Chat Memory is ready. Ask a question below (type 'exit' to quit):")

while True:
    query = input("\n🔎 Your question: ")
    if query.lower() == "exit":
        break

    try:
        result = qa_chain.invoke({"question": query})
        print(f"\n🤖 Answer:\n{result['answer']}")

        print("\n📂 Source Documents:")
        # Check if 'source_documents' exists and is not empty before iterating
        if "source_documents" in result and result["source_documents"]:
            for doc in result["source_documents"]:
                # Safely get the 'source' metadata, defaulting to 'unknown'
                source = doc.metadata.get("source", "unknown")
                # Ensure page_content is a string and slice it
                content_preview = doc.page_content.strip()[:300] if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) else "Content unavailable"
                print(f"→ Source: {source}\nContent Preview:\n{content_preview}\n---")
        else:
            print("No source documents found for this query.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try your question again or check your configuration.")
