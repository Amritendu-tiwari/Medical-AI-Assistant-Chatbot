import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Try modern LangChain imports first, fallback where needed
try:
    from langchain.embeddings import OpenAIEmbeddings
except Exception:
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None

try:
    from langchain.vectorstores import FAISS
except Exception:
    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        FAISS = None

# Retriever documents
try:
    from langchain.schema import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        Document = None

if FAISS is None:
    raise ImportError("FAISS vectorstore not available. Install langchain and langchain_community or upgrade packages.")

# load .env if present (will not overwrite existing env vars)
load_dotenv(".env")

def _get_openai_embeddings():
    """Instantiate OpenAIEmbeddings and ensure API key is present."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Set it in the environment or place it in ../.env (and ensure load_dotenv can read it)."
        )
    if OpenAIEmbeddings is None:
        raise ImportError("OpenAIEmbeddings import failed. Install 'langchain' or 'langchain_openai'.")
    # Instantiate the embeddings object; some langchain versions accept openai_api_key param
    try:
        emb = OpenAIEmbeddings(openai_api_key=api_key)
    except TypeError:
        # fallback to other constructor signatures
        emb = OpenAIEmbeddings()
        # some versions read key from env, so it's OK
    return emb

def get_retriever(vector_db_path: str = "faiss_index", top_k: int = 5, allow_reload: bool = True):
    """
    Load FAISS vectorstore from `vector_db_path` and return a retriever configured with k=top_k.

    Args:
        vector_db_path: path to the FAISS index directory (where you saved it).
        top_k: number of top results to return from retriever.
        allow_reload: if True, tries to load index even if the embeddings param isn't accepted by load_local.

    Returns:
        a LangChain retriever (object with get_relevant_documents(query) method).
    """
    vector_db_path = str(vector_db_path)
    if not Path(vector_db_path).exists():
        raise FileNotFoundError(f"FAISS index not found at {vector_db_path}. Please run your embedding/index creation script first.")

    # instantiate embeddings (used for loading index in some langchain versions)
    embeddings = None
    try:
        embeddings = _get_openai_embeddings()
    except Exception as e:
        # we won't fail immediately because some FAISS.load_local variants don't require embeddings
        print(f"[get_retriever] Warning: could not instantiate embeddings: {e}. Will still attempt to load index without embeddings.")

    # Try the most compatible load variants
    vectorstore = None
    load_errors = []
    try:
        # common modern signature: FAISS.load_local(folder_path, embeddings)
        if embeddings is not None:
            vectorstore = FAISS.load_local(vector_db_path, embeddings)
        else:
            # try loading without embeddings
            vectorstore = FAISS.load_local(vector_db_path)
    except Exception as e:
        load_errors.append(e)
        # try fallback: allow_dangerous_deserialization flag used in some older versions
        try:
            if embeddings is not None:
                vectorstore = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
            else:
                vectorstore = FAISS.load_local(vector_db_path, allow_dangerous_deserialization=True)
        except Exception as e2:
            load_errors.append(e2)
            vectorstore = None

    if vectorstore is None:
        # if still None, raise a consolidated error
        err_msg = "Failed to load FAISS index. Errors:\n" + "\n".join([str(e) for e in load_errors])
        raise RuntimeError(err_msg)

    print(f"[get_retriever] FAISS index loaded from {vector_db_path}")

    # create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever

# If executed as script, run quick demo queries
if __name__ == "__main__":
    try:
        retriever = get_retriever(vector_db_path="faiss_index", top_k=5)
    except Exception as e:
        print("Error creating retriever:", e)
        sys.exit(1)

    test_queries = [
        "How to prevent viral fever",
        "What is vitamin B12",
        "Foods high in B12"
    ]

    for q in test_queries:
        print(f"\n\n=== QUERY: {q} ===")
        try:
            docs = retriever.get_relevant_documents(q)
        except AttributeError:
            # older/newer mismatch: try .retrieve(...) or .get_relevant_documents
            try:
                docs = retriever.retrieve(q)
            except Exception as e:
                print("Retriever does not support get_relevant_documents or retrieve: ", e)
                docs = []
        if not docs:
            print("No documents retrieved.")
            continue
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
            page = doc.metadata.get("page") if hasattr(doc, "metadata") else None
            print(f"--- Result {i} ---")
            print("Source:", src)
            print("Page:", page)
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            print(content[:400].strip().replace("\n", " "))
            print("...")

