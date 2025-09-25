import os
import sys
import json
import math
from pathlib import Path
from typing import List, Sequence, Any

from dotenv import load_dotenv

# LangChain imports
from langchain.schema import Document
from langchain.vectorstores import FAISS
# Try modern embeddings import, fallback to older if needed
try:
    from langchain.embeddings import OpenAIEmbeddings
except Exception:
    from langchain_openai import OpenAIEmbeddings

# load .env if present
load_dotenv(".env")

# ---------- CONFIG ----------
CHUNKS_FILE_DEFAULT = "data/Processed_data/processed_chunks.json"
FAISS_DIR_DEFAULT = "faiss_index"
# Keep a safe margin below OpenAI provider limit (300k). Use 280k to be safe.
MAX_TOKENS_PER_REQUEST = 280_000
# Fallback approx token size per word (when tiktoken isn't available)
APPROX_TOKENS_PER_WORD = 1.33
# Default batch max docs (absolute upper bound to avoid giant batch counts)
ABSOLUTE_MAX_DOCS_PER_BATCH = 512
# --------------------------------

# Try to import tiktoken for accurate tokenization
try:
    import tiktoken
    TIKTOKEN_AVAIL = True
    # choose encoding consistent with OpenAI modern models
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    TIKTOKEN_AVAIL = False
    _ENC = None


def count_tokens_for_text(text: str) -> int:
    """Return token count for a text. Uses tiktoken if available, else approximates by words."""
    if TIKTOKEN_AVAIL and _ENC is not None:
        return len(_ENC.encode(text))
    # approximate: number of words * factor
    words = len(text.split())
    return math.ceil(words * APPROX_TOKENS_PER_WORD)


class BatchedOpenAIEmbeddings:
    """
    Wrapper around OpenAIEmbeddings that batches embed_documents calls
    so that total tokens per request don't exceed MAX_TOKENS_PER_REQUEST.
    It exposes the same interface used by LangChain: embed_documents and embed_query.
    """

    def __init__(self, openai_embeddings: OpenAIEmbeddings,
                 max_tokens_per_request: int = MAX_TOKENS_PER_REQUEST,
                 absolute_max_docs: int = ABSOLUTE_MAX_DOCS_PER_BATCH):
        self.inner = openai_embeddings
        self.max_tokens_per_request = max_tokens_per_request
        self.absolute_max_docs = absolute_max_docs

        # Try to detect if inner has an embed_documents method
        if not hasattr(self.inner, "embed_documents"):
            raise ValueError("Provided embeddings object must implement embed_documents(texts: List[str])")

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Batch texts so each underlying request has total tokens <= max_tokens_per_request.
        Returns list of embedding vectors in the same order as texts.
        """
        embeddings: List[List[float]] = []
        batch_texts: List[str] = []
        batch_tokens = 0
        docs_in_batch = 0

        def flush_batch():
            nonlocal batch_texts, batch_tokens, docs_in_batch, embeddings
            if not batch_texts:
                return
            # call underlying embed_documents for this batch
            vectors = self.inner.embed_documents(batch_texts)
            embeddings.extend(vectors)
            # reset
            batch_texts = []
            batch_tokens = 0
            docs_in_batch = 0
            return

        for txt in texts:
            txt_tokens = count_tokens_for_text(txt)
            # If a single doc exceeds the max tokens, we still must send it alone.
            # Warn the user (print) in that case.
            if txt_tokens >= self.max_tokens_per_request:
                print(f"[make_faiss] WARNING: single document token size {txt_tokens} >= max_tokens_per_request {self.max_tokens_per_request}. "
                      f"Embedding it in a single request (may still fail).")
                # flush current batch first
                flush_batch()
                # send this large doc alone
                vectors = self.inner.embed_documents([txt])
                embeddings.extend(vectors)
                continue

            # if adding this doc would exceed token budget or absolute docs cap, flush
            if (batch_tokens + txt_tokens > self.max_tokens_per_request) or (docs_in_batch + 1 > self.absolute_max_docs):
                flush_batch()

            # add to current batch
            batch_texts.append(txt)
            batch_tokens += txt_tokens
            docs_in_batch += 1

        # flush leftovers
        flush_batch()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # embed a single query via underlying object (small request)
        return self.inner.embed_query(text)


def load_chunks(chunks_file: str) -> List[Document]:
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Chunks JSON must be a list of chunk dicts")

    docs: List[Document] = []
    for i, c in enumerate(data):
        text = c.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        meta = {
            "source": c.get("source"),
            "chunk_id": c.get("chunk_id"),
            "page": c.get("page"),
            "start_index": c.get("start_index"),
            "end_index": c.get("end_index"),
            "num_tokens": c.get("num_tokens")
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def get_openai_embeddings_obj():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment. Set it or place it in ../.env and run again.")
    # create LangChain OpenAIEmbeddings object
    emb = OpenAIEmbeddings(openai_api_key=api_key)
    return emb


def build_faiss_from_docs(docs: List[Document], faiss_dir: str, overwrite: bool = False):
    if len(docs) == 0:
        raise ValueError("No documents provided to build FAISS index.")

    base_emb = get_openai_embeddings_obj()
    batched_emb = BatchedOpenAIEmbeddings(base_emb, max_tokens_per_request=MAX_TOKENS_PER_REQUEST)

    # Use langchain FAISS.from_documents which will call our batched embedder
    print(f"[make_faiss] Building FAISS index from {len(docs)} documents. This will call embeddings in batches.")
    vectorstore = FAISS.from_documents(docs, batched_emb)

    # Save index
    outp = Path(faiss_dir)
    if outp.exists() and any(outp.iterdir()):
        if overwrite:
            # remove previous contents
            import shutil
            shutil.rmtree(outp)
            outp.mkdir(parents=True, exist_ok=True)
        else:
            # attempt to load and merge: easier to overwrite to avoid complexity
            print(f"[make_faiss] FAISS directory {faiss_dir} already exists. Overwriting by default (use --overwrite to control).")
            import shutil
            shutil.rmtree(outp)
            outp.mkdir(parents=True, exist_ok=True)

    os.makedirs(faiss_dir, exist_ok=True)
    vectorstore.save_local(faiss_dir)
    print(f"[make_faiss] FAISS index saved to {faiss_dir}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    overwrite = "--overwrite" in sys.argv

    chunks_file = args[0] if len(args) >= 1 else CHUNKS_FILE_DEFAULT
    faiss_dir = args[1] if len(args) >= 2 else FAISS_DIR_DEFAULT

    print("[make_faiss] chunks_file:", chunks_file)
    print("[make_faiss] faiss_dir:", faiss_dir)
    print("[make_faiss] overwrite:", overwrite)
    print("[make_faiss] Tiktoken available:", TIKTOKEN_AVAIL)
    try:
        docs = load_chunks(chunks_file)
    except Exception as e:
        print("[make_faiss] ERROR loading chunks:", e)
        sys.exit(1)

    print(f"[make_faiss] Loaded {len(docs)} chunk documents. Estimating tokens...")

    # optional: quick summary token estimate per document and total (approx)
    total_est = 0
    for d in docs[:50]:  # sample first 50 for an estimate (fast)
        t = count_tokens_for_text(d.page_content)
        total_est += t
    print(f"[make_faiss] Sampled token estimate (first 50 docs): {total_est} tokens. (Estimates only)")

    try:
        build_faiss_from_docs(docs, faiss_dir, overwrite=overwrite)
    except Exception as e:
        print("[make_faiss] ERROR building FAISS:", e)
        raise


if __name__ == "__main__":
    main()
