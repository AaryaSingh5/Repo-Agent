# Code Repository Q&A Agent - Walkthrough

I have successfully built the code repository Q&A agent based on the architecture you provided. Here is a breakdown of what was created and how you can use it.

## What was built

I created a Python project in your `New folder` with the following components:

1.  [`requirements.txt`](file:///c:/Users/Intel/OneDrive/Desktop/New%20folder/requirements.txt): Contains all the necessary libraries (`langchain`, `faiss-cpu`, `sentence-transformers`, etc.).
2.  [`.env.example`](file:///c:/Users/Intel/OneDrive/Desktop/New%20folder/.env.example): A template for your environment variables.
3.  [`ingest.py`](file:///c:/Users/Intel/OneDrive/Desktop/New%20folder/ingest.py): The script that loads your code repository, splits the code into chunks, generates vector embeddings locally using `sentence-transformers`, and saves them into a local FAISS database.
4.  [`agent.py`](file:///c:/Users/Intel/OneDrive/Desktop/New%20folder/agent.py): The interactive query script. It loads the FAISS index, initializes the free Hugging Face LLM API (now using the extremely capable `Qwen/Qwen2.5-72B-Instruct` via the robust `huggingface_hub.InferenceClient`), and lets you ask questions about the indexed codebase.

> [!NOTE]  
> The dependencies have already been installed in your main Python 3.13 environment.
> We switched from AST-parsing to standard Text Splitting because the `tree_sitter` language bindings do not yet have pre-built wheels for Python 3.13. This fallback method is highly robust and standard for RAG pipelines.

## How to use the Agent

Follow these steps to run the agent on your own code:

### Step 1: Set up your API Token
1. Go to your `New folder` and rename `.env.example` to `.env`.
2. Open the `.env` file and replace `your_token_here` with your free Hugging Face Access Token. You can generate one for free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Step 2: Ingest a Repository
Run the ingest script in your terminal to build the vector database. By default, it ingests the current directory:
```bash
python ingest.py
```
If you want to ingest a different repository, you can pass the path:
```bash
python ingest.py --repo "C:\path\to\your\other\repo"
```
This will create a folder named `faiss_index` containing the vector database.

### Step 3: Ask Questions
Start the interactive Q&A agent:
```bash
python agent.py
```

> [!TIP]  
> You can launch the agent with multiple repositories simultaneously using the `--repos` argument! You can pass local paths or GitHub shorthands (e.g., `user/repo`):
> ```bash
> python agent.py --repos psf/requests tiangolo/fastapi
> ```
> This will automatically clone, index, and merge all of them into a single knowledge base!

> [!IMPORTANT]
> You can also index an entire GitHub account! Use the `--github-user` argument to fetch, clone, and build a unified knowledge base across all repositories for a user:
> ```bash
> python agent.py --github-user AaryaSingh5
> ```
> *Note: This will clone all public repositories. To access private repositories, add a `GITHUB_TOKEN` to your `.env` file.*

Once it initializes, you can type your questions (e.g., "What does the ingest script do?"). The agent will retrieve the relevant code snippets and generate an answer using the free Hugging Face API!

You can also dynamically add new repositories to the active knowledge base at any time by using the `/add` command in the chat:
```text
You: /add psf/requests
```
The agent will instantly download, parse, and merge the new repository's embeddings into your active session without losing context!

> [!TIP]  
> If you ever update your code, simply run `python ingest.py` again to refresh the vector database.
