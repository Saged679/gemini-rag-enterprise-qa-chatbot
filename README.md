# 💬 RAG-Based Enterprise Policy Chatbot

A domain-specific Question Answering chatbot that uses **Retrieval-Augmented Generation (RAG)** with **Google Gemini**, **ChromaDB**, and **LangChain** to answer questions from HR, IT Security, and Code of Conduct policies.

🎓 Designed as a **production-grade student showcase project** for real-world enterprise NLP use cases.

---

## ✨ Key Features

* 📄 Multi-file PDF upload (HR / IT / Conduct policies)
* 🔍 Text chunking + embedding with Gemini Embedding API
* 🧠 Vector search via ChromaDB
* 🤖 Question Answering with Gemini 1.5 Flash
* 🧠 Chat Memory (follow-up question support)
* 📚 Source document highlighting in answers
* 💬 Streamlit chatbot with polished UI
* ⚙️ FastAPI backend for API integration

---

## 🧱 Tech Stack

| Tool / Library         | Purpose                            |
| ---------------------- | ---------------------------------- |
| `LangChain`            | RAG pipeline & orchestration       |
| `Google Generative AI` | Embeddings + LLM Q\&A (Gemini)     |
| `ChromaDB`             | Local vector database (FAISS-like) |
| `Streamlit`            | Web-based chat UI                  |
| `FastAPI`              | JSON API backend                   |
| `PyMuPDF`              | PDF parsing & text extraction      |
| `.env` + `dotenv`      | Secret management (API key)        |

---

## 🗂️ Project Structure

```
rag-enterprise-policy-chatbot/
├── data/
│   └── raw_docs/             # HR/IT/Code of Conduct PDFs
├── vectorstore/
│   └── chroma/               # Stored document embeddings
├── app.py                    # Streamlit chatbot UI
├── main.py                   # FastAPI API server
├── embed_and_store.py        # One-time PDF chunking + vector DB build
├── requirements.txt          # Project dependencies
├── .env.example              # API key format
├── README.md                 # You’re reading it!
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-enterprise-policy-chatbot.git
cd rag-enterprise-policy-chatbot
```

### 2. Add Your Gemini API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

➡️ Get a key from: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

---

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ➤ Step 1: Preprocess & Index PDFs

Put your PDFs in the `data/raw_docs/` folder (e.g., HR, IT, conduct policies).

Then run:

```bash
python embed_and_store.py
```

### ➤ Step 2: Launch Chatbot (Streamlit)

```bash
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

### ➤ Step 3: API Mode (FastAPI)

```bash
uvicorn main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💬 Example Questions

* What is the company’s sick leave policy?
* How do I report workplace harassment?
* What is the password expiration rule?
* Who do I contact in case of misconduct?

---

## 📦 API Endpoint Example (FastAPI)

```http
POST /query

Request Body:
{
  "question": "What is the vacation policy?"
}

Response:
{
  "answer": "...",
  "sources": ["source chunk 1...", "source chunk 2..."]
}
```

---

## 🧠 Chat Memory

Supports follow-up questions in context, e.g.:

```
User: What is the sick leave policy?
Bot: Employees are entitled to X days...

User: What about for part-time workers?
Bot: Part-time workers receive Y days of sick leave...
```

---

## 👨‍💻 Author

**Saged Ahmed**
AI/ML Engineer & Student

---

## 📌 License

This project is for educational purposes. No proprietary company documents are used.
