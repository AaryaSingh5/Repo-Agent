import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
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

    # 1. Load the embedding model via Free API (must match the one used during ingestion)
    print("Loading embedding model via Free API...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

    # 2. Load the vector store
    print("Loading FAISS vector database...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 3. Initialize the LLM (using Hugging Face Inference API)
    # We use a free-tier compatible model like mistralai/Mistral-7B-Instruct-v0.2
    # Ensure HUGGINGFACEHUB_API_TOKEN is set in your .env file
    print("Initializing LLM via Hugging Face API...")
    # Using HuggingFaceEndpoint for the Serverless Inference API
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token
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

def interactive_session(index_path: str = "faiss_index"):
    """
    Starts an interactive Q&A loop.
    """
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
    
    args = parser.parse_args()
    interactive_session(args.index)
