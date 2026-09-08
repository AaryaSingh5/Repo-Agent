# 🤖 Code Repository Q&A Agent

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-orange)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-lightgrey)

**Code Repository Q&A Agent** is an AI-powered developer tool that allows you to "talk" to your codebase. By leveraging the power of Retrieval-Augmented Generation (RAG), local vector embeddings, and free Hugging Face language models, this agent ingests your entire repository and accurately answers architectural, structural, and implementation questions about your code.

---

## ✨ Features

- **Any Repository:** Point the agent at any local folder or repository, and it will automatically ingest all supported text and source code files.
- **Privacy-First Embeddings:** Vector embeddings are generated 100% locally on your machine using `sentence-transformers`, meaning your raw code is never sent to a third-party embedding API.
- **Cost-Free LLM Inference:** Powered by the free Hugging Face Inference API, utilizing state-of-the-art open-weight models (like Mistral or Zephyr) without requiring an expensive API subscription or local GPU.
- **Robust Parsing:** Automatically handles various file extensions, detects encodings, and splits code into semantic chunks optimized for context window limits.

---

## 🧠 Architecture Overview

This project implements a classic RAG (Retrieval-Augmented Generation) pipeline:

1. **Ingestion (`DirectoryLoader` & `TextLoader`):** Scans the target directory, excluding virtual environments and hidden git files, and loads the raw text of your source code.
2. **Chunking (`RecursiveCharacterTextSplitter`):** Breaks large source files down into smaller, overlapping chunks (1000 characters by default) to preserve context.
3. **Embedding (`HuggingFaceEmbeddings`):** Uses the local `all-MiniLM-L6-v2` model to convert these text chunks into high-dimensional vectors.
4. **Vector Storage (`FAISS`):** Stores these vectors in a local, highly efficient FAISS index (`faiss_index/`).
5. **Retrieval & Generation (`agent.py`):** When you ask a question, the agent embeds your query, retrieves the top 5 most relevant code chunks from FAISS, and passes them as context to a Hugging Face LLM to generate a precise answer.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+**
- Git

You will also need a **Free Hugging Face Access Token**.
1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Go to your [Settings -> Access Tokens](https://huggingface.co/settings/tokens) and generate a new token (Read access is sufficient).

---

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AaryaSingh5/Repo-Agent.git
   cd Repo-Agent
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file from the provided template:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and insert your Hugging Face API token:
   ```env
   HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
   ```

---

## 📖 Usage

Using the agent is a two-step process: First, you index the codebase, and second, you run the query agent.

### Step 1: Ingest the Codebase

Run the `ingest.py` script. By default, it will ingest the current directory (`.`):

```bash
python ingest.py
```

**Ingesting a different repository:**
You can point the ingestion script at any folder on your machine using the `--repo` flag:
```bash
python ingest.py --repo "/path/to/another/project" --index "my_custom_index"
```

*Note: The first time you run this, it will download the `sentence-transformers` embedding model (~90MB) to your machine.*

### Step 2: Talk to your Code

Once the FAISS index is created, start the interactive chat session:

```bash
python agent.py
```

If you saved your index under a custom name, specify it:
```bash
python agent.py --index "my_custom_index"
```

**Example Interaction:**
```
🤖 Code Repository Q&A Agent Initialized!
Ask any question about your codebase.
Type 'exit' or 'quit' to stop.
==================================================

You: What does the ingest.py script do?

Thinking...

🤖 Agent:
Based on the provided context, the `ingest.py` script is responsible for the following:
1. It uses a `DirectoryLoader` to scan and load text files from a specified repository path, excluding folders like `node_modules`, `venv`, and `.git`.
2. It splits the loaded documents into smaller chunks using a `RecursiveCharacterTextSplitter`.
3. It generates embeddings for these chunks using a local `HuggingFaceEmbeddings` model (`all-MiniLM-L6-v2`).
4. Finally, it builds a FAISS vector database from these embeddings and saves it locally to a specified path (default is `faiss_index`).

[Sources referenced:]
  1. C:\Users\...\Repo-Agent\ingest.py
  2. C:\Users\...\Repo-Agent\ingest.py
```

---

## ⚙️ Configuration & Customization

You can easily modify the scripts to fit your specific needs:
- **Change the LLM:** In `agent.py`, change `repo_id="mistralai/Mistral-7B-Instruct-v0.2"` to any other free Instruct model on the Hugging Face Hub (e.g., `HuggingFaceH4/zephyr-7b-beta`).
- **Change the Chunk Size:** In `ingest.py`, modify `chunk_size` and `chunk_overlap` in the `RecursiveCharacterTextSplitter` initialization to fine-tune how much code is sent in a single context block.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a Pull Request if you'd like to improve the parsers, add UI components, or support local LLM inference engines like Ollama.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
