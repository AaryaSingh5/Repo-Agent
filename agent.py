import os
import argparse
import subprocess
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import ingest_repository
# Removed legacy chains imports

def load_agent(index_path: str = "faiss_index"):
    """
    Loads the FAISS index and initializes the Hugging Face LLM and RAG chain.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run ingest.py first.")

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token or hf_token == "your_token_here":
        raise ValueError("Please set a valid HUGGINGFACEHUB_API_TOKEN in the .env file.")

    # 1. Load the embedding model (local)
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2) locally...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Load the vector store
    print("Loading FAISS vector database...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 3. Initialize the LLM (using Hugging Face Inference API)
    # We use a free-tier compatible model like mistralai/Mistral-7B-Instruct-v0.2
    # Ensure HUGGINGFACEHUB_API_TOKEN is set in your .env file
    print("Initializing LLM via Hugging Face API...")
    
    from huggingface_hub import InferenceClient
    from langchain_core.language_models.llms import LLM
    from typing import Optional, List, Any

    class HuggingFaceChatLLM(LLM):
        repo_id: str
        api_token: str
        temperature: float = 0.1
        max_tokens: int = 512
        
        @property
        def _llm_type(self) -> str:
            return "huggingface_chat"
            
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            client = InferenceClient(model=self.repo_id, token=self.api_token)
            messages = [{"role": "user", "content": prompt}]
            try:
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                raise Exception(f"HF API Error: {e}")

    llm = HuggingFaceChatLLM(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        api_token=hf_token
    )

    # 4. Create a custom Retrieval Chain logic to avoid dependency issues
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    system_prompt = (
        "You are an expert AI coding assistant. Use the following pieces of retrieved code context "
        "to answer the user's question about the repository. "
        "If you don't know the answer or the context doesn't contain the answer, just say that you don't know. "
        "Do not make up an answer. Provide clear, concise explanations and code examples if appropriate.\n\n"
        "Context:\n{context}\n\n"
        "User Question: {question}"
    )

    class CustomRAGChain:
        def __init__(self, retriever, llm, prompt_template):
            self.retriever = retriever
            self.llm = llm
            self.prompt_template = prompt_template

        def invoke(self, inputs):
            query = inputs["input"]
            docs = self.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            final_prompt = self.prompt_template.format(context=context, question=query)
            answer = self.llm.invoke(final_prompt)
            
            return {
                "answer": answer,
                "context": docs
            }

    rag_chain = CustomRAGChain(retriever, llm, system_prompt)
    return rag_chain

def clone_repo(repo_url: str, base_dir: str = "repos") -> str:
    """Clones a single repository and returns the local path. Supports GitHub shorthand."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Expand github shorthand if it's "user/repo"
    if "/" in repo_url and not repo_url.startswith(("http://", "https://", "git@")) and not os.path.exists(repo_url):
        if len(repo_url.split("/")) == 2:
            repo_url = f"https://github.com/{repo_url}.git"

    if repo_url.startswith(("http://", "https://", "git@")):
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        local_repo_path = os.path.join(base_dir, repo_name)
        if not os.path.exists(local_repo_path):
            print(f"Cloning {repo_url} into {local_repo_path}...")
            subprocess.run(["git", "clone", repo_url, local_repo_path], check=True)
        else:
            print(f"Repository already cloned at {local_repo_path}")
        return local_repo_path
    else:
        if os.path.exists(repo_url):
            return repo_url
        else:
            raise ValueError(f"Invalid repository path or URL: {repo_url}")

def prepare_repos(repo_urls: list) -> tuple[str, str]:
    """Prepares multiple repositories by cloning and returning the local base path and index path."""
    if len(repo_urls) == 1:
        local_repo_path = clone_repo(repo_urls[0], base_dir="repos")
        repo_name = os.path.basename(os.path.abspath(local_repo_path))
        index_path = f"{repo_name}_faiss_index"
        
        if not os.path.exists(index_path):
            print(f"Index '{index_path}' not found. Running ingestion for {local_repo_path}...")
            ingest_repository(local_repo_path, save_path=index_path)
            
        return local_repo_path, index_path
    else:
        base_dir = "repos/multi_repo"
        for repo_url in repo_urls:
            clone_repo(repo_url, base_dir=base_dir)
            
        index_path = "multi_repo_faiss_index"
        if not os.path.exists(index_path):
            print(f"Index '{index_path}' not found. Running ingestion for {base_dir}...")
            ingest_repository(base_dir, save_path=index_path)
            
        return base_dir, index_path

def prepare_github_user(username: str) -> tuple[str, str]:
    """Fetches, clones, and prepares an index for all repositories of a GitHub user."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token and token != "your_token_here":
        headers["Authorization"] = f"token {token}"
        
    print(f"Fetching repositories for user/org {username}...")
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code} - {response.text}")
            break
        data = response.json()
        if not data:
            break
        repos.extend(data)
        page += 1
        
    if not repos:
        raise ValueError(f"No repositories found for {username}")
        
    print(f"Found {len(repos)} repositories.")
    for i, repo in enumerate(repos):
        print(f"[{i+1}] {repo['name']}")
    
    print("\nEnter the numbers of the repositories you want to index (comma-separated), or press Enter for all:")
    selection = input("> ").strip()
    if selection:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            repos = [repos[i] for i in indices if 0 <= i < len(repos)]
            print(f"Selected {len(repos)} repositories.")
        except Exception as e:
            print("Invalid selection, proceeding with all repositories.")
    base_dir = os.path.join("repos", username)
    os.makedirs(base_dir, exist_ok=True)
    
    for repo in repos:
        repo_name = repo["name"]
        clone_url = repo["clone_url"]
        local_repo_path = os.path.join(base_dir, repo_name)
        
        if not os.path.exists(local_repo_path):
            print(f"Cloning {repo_name}...")
            subprocess.run(["git", "clone", clone_url, local_repo_path], check=False)
        else:
            print(f"Repository {repo_name} already cloned.")
            
    index_path = f"{username}_faiss_index"
    if not os.path.exists(index_path):
        print(f"Index '{index_path}' not found. Running unified ingestion for all repositories of {username}...")
        ingest_repository(base_dir, save_path=index_path)
        
    return base_dir, index_path

def interactive_session(index_path: str = "faiss_index", repos: list = None, github_user: str = None):
    """
    Starts an interactive Q&A loop.
    """
    if github_user:
        try:
            _, index_path = prepare_github_user(github_user)
        except Exception as e:
            print(f"Failed to prepare github user repositories: {e}")
            return
    elif repos:
        try:
            _, index_path = prepare_repos(repos)
        except Exception as e:
            print(f"Failed to prepare repositories: {e}")
            return

    try:
        rag_chain = load_agent(index_path)
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    print("\n" + "="*50)
    print("🤖 Code Repository Q&A Agent Initialized!")
    print("Ask any question about your codebase.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue

        if query.startswith("/add "):
            repo_to_add = query.split(" ", 1)[1].strip()
            print(f"\nAdding repository: {repo_to_add}")
            try:
                local_repo_path = clone_repo(repo_to_add, base_dir="repos")
                print("Ingesting new repository...")
                new_vectorstore = ingest_repository(local_repo_path, save_path=None)
                if new_vectorstore:
                    rag_chain.retriever.vectorstore.merge_from(new_vectorstore)
                    rag_chain.retriever.vectorstore.save_local(index_path)
                    print("Successfully merged and updated index.")
                else:
                    print("No valid documents found in the new repository.")
            except Exception as e:
                print(f"Failed to add repository: {e}")
            continue

        print("\nThinking...")
        try:
            # Invoke the chain
            response = rag_chain.invoke({"input": query})
            
            # Print the answer
            print("\n🤖 Agent:")
            print(response["answer"])
            
            # Print the sources used
            print("\n[Sources referenced:]")
            for i, doc in enumerate(response["context"]):
                source = doc.metadata.get("source", "Unknown")
                print(f"  {i+1}. {source}")
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"\nError generating answer: {e}\n")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Query the ingested code repository.")
    parser.add_argument("--index", type=str, default="faiss_index", help="Path to the saved FAISS index.")
    parser.add_argument("--repos", nargs="+", type=str, help="One or more paths or URLs to the repositories to analyze.")
    parser.add_argument("--github-user", type=str, help="GitHub username to ingest all repositories for.")
    
    args = parser.parse_args()
    interactive_session(args.index, args.repos, args.github_user)
