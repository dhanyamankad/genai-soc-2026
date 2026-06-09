# 🚀 GENAI-SOC 2026 Journey
Welcome to my repository for the MSTC Summer of Code 2026 – Generative AI Track.
This repository contains my weekly progress, projects, experiments, and learnings throughout the program.

---

## 👩‍💻 About Me
Hi, I'm Dhanya Mankad, an engineering student passionate about:
- Artificial Intelligence
- Software Development
- Problem Solving
- Building Real-World Projects

This repository documents my journey of learning and building with Generative AI.

---

## 📅 Week 0 — Environment Setup & Foundations

### Objectives Completed
- ✅ Installed Python and VS Code
- ✅ Configured Python Extension
- ✅ Created and managed Virtual Environments
- ✅ Learned Git & GitHub basics
- ✅ Set up Google Colab
- ✅ Created GitHub Repository
- ✅ Generated and tested Groq API Key
- ✅ Successfully made first API call

---

## 🤖 Week 1 — PromptForge: Multi-Persona AI Assistant

### Project Overview
PromptForge is a multi-persona AI assistant built using Python, Groq, and Gradio.
The application demonstrates core prompt engineering techniques by allowing users to switch between different AI personalities, each with its own behavior, few-shot examples, and response style.

👉 [View Project](./week1-promptforge)

### ✨ Features

🔹 **Technical Explainer**
Explains technical concepts clearly using analogies and simple language.

🔹 **Debate Coach**
Presents balanced arguments on both sides of any topic.

🔹 **Code Reviewer**
Analyzes code and returns structured JSON feedback with issues, suggestions, and severity.

🔹 **Creative Writer**
Generates vivid, cinematic narrative prose.

---

## 📚 Week 2 — DocBuddy Pro: Q&A Over Multiple PDFs with Source Citations

### Project Overview
DocBuddy Pro is a full RAG (Retrieval-Augmented Generation) pipeline built from scratch using Python, LangChain, ChromaDB, and Gradio. Upload multiple PDFs and ask questions across all of them — every answer cites the exact source document and page number. A collapsible panel shows exactly which chunks were retrieved, making the RAG process fully transparent.

👉 [View Project](./week2-docbuddy)

### 🧠 Concepts Implemented
- **Chunking** — `RecursiveCharacterTextSplitter` splits PDF pages into 500-char overlapping chunks
- **Embeddings** — `all-MiniLM-L6-v2` (HuggingFace) converts each chunk into a 384-dimensional vector
- **Vector Database** — ChromaDB stores and retrieves vectors with metadata (filename + page number)
- **Full RAG Pipeline** — question → embed → similarity search → grounded prompt → answer
- **Anti-Hallucination** — model refuses to answer anything not found in the uploaded documents
- **Source Citations** — every answer cites `[Source: filename, Page X]`
- **Multi-Document RAG** — correctly retrieves and cites across multiple PDFs simultaneously

### ✨ Features

🔹 **Multi-PDF Upload**
Upload multiple PDFs at once — lecture notes, research papers, policy documents.

🔹 **Persistent Indexing**
ChromaDB saves to disk — no re-indexing needed on restart.

🔹 **Grounded Answers with Citations**
Every response includes the source filename and page number.

🔹 **Anti-Hallucination**
Tested with out-of-scope questions — model correctly says "I don't have that information in the provided documents."

🔹 **Retrieved Context Panel**
Collapsible accordion shows the exact chunks used to generate each answer — great for understanding how RAG works under the hood.

### 🛠 Tech Stack
| Tool | Purpose |
|---|---|
| Gradio | Web UI |
| PyPDF | PDF text extraction |
| LangChain | RAG pipeline orchestration |
| HuggingFace `all-MiniLM-L6-v2` | Local embeddings |
| ChromaDB | Vector database |
| Groq `llama-3.1-8b-instant` | LLM for answer generation |
