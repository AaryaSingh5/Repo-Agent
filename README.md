# Code Repository Q&A Agent

An AI agent that can ingest any local code repository, index its contents, and answer questions using free Hugging Face models.

## Architecture

This project uses a standard Retrieval-Augmented Generation (RAG) pipeline tailored for codebases:
1. **Loader & Splitter**: Parses source code files into chunks.
2. **Embedding Model**: Uses a local model (`sentence-transformers/all-MiniLM-L6-v2`) to convert code chunks into vector representations.
3. **Vector Database**: Stores embeddings locally using `FAISS`.
4. **LLM Generator**: Connects to the free Hugging Face Inference API (`Mistral-7B-Instruct-v0.2`) to generate answers based on retrieved context.

## Requirements

Ensure you have Python installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the root directory by copying the provided template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your free Hugging Face API token (`HUGGINGFACEHUB_API_TOKEN`). You can get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

### 1. Ingest a Repository
Run the ingest script to build the vector database. By default, it ingests the current directory:

```bash
python ingest.py
```

To ingest a specific repository, provide the `--repo` flag:

```bash
python ingest.py --repo "C:\path\to\your\repo"
```

This will create a `faiss_index` folder containing the local vector database.

### 2. Ask Questions
Start the interactive Q&A agent:

```bash
python agent.py
```

Type your questions about the codebase. The agent will retrieve the relevant code snippets from the FAISS database and generate an answer using the Hugging Face API!
