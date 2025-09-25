import os
import json
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Callable, Dict, Any

# LangChain imports (modern package names)
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # kept for fallback-ish usage
from langchain.schema import Document

# Try tiktoken for accurate token counts (preferred)
try:
    import tiktoken
    TOKT_AVAILABLE = True
except Exception:
    tiktoken = None
    TOKT_AVAILABLE = False

# ---------------- CONFIG ----------------
RAW_DIR = "data/Raw_data"               # keep path unchanged per your request
OUT_DIR = "data/Processed_data"         # keep path unchanged per your request
OUT_CHUNKS = "processed_chunks.json"
OUT_METADATA = "ingestion_metadata.json"

CHUNK_TARGET = 500  # target token count (in 400-600 range)
CHUNK_OVERLAP = 100
HEADER_FOOTER_TOP_LINES = 3
HEADER_FOOTER_BOT_LINES = 3
REPEAT_THRESHOLD = 0.35
# ----------------------------------------

# small utilities
def clean_whitespace(text: str) -> str:
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

PAGE_NOISE_PATTERNS = [
    r'^\s*Page\s*\d+\s*$',
    r'^\s*\d+\s*$',
    r'^\s*Page\s*\d+\s*of\s*\d+\s*$',
    r'^\s*-\s*\d+\s*-\s*$',
]
PAGE_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in PAGE_NOISE_PATTERNS]

def looks_like_page_number(line: str) -> bool:
    if not line:
        return False
    if len(line) > 40 and len(line.split()) > 6:
        return False
    for rx in PAGE_NOISE_RE:
        if rx.match(line.strip()):
            return True
    return False

# Tokenizer abstraction
def get_tokenizer():
    if TOKT_AVAILABLE and tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        def tokenize(text: str):
            return enc.encode(text)
        def detokenize(tokens):
            return enc.decode(tokens)
        def count_tokens(text: str):
            return len(enc.encode(text))
        return tokenize, detokenize, count_tokens, "tiktoken_cl100k_base"

    # fallback: whitespace-based approx
    def tokenize_ws(text: str):
        return text.split()
    def detokenize_ws(tokens):
        return " ".join(tokens)
    def count_tokens_ws(text: str):
        return len(text.split())
    return tokenize_ws, detokenize_ws, count_tokens_ws, "whitespace_approx"

# header/footer detection & stripping
def detect_repeated_header_footer(pages: List[Document]) -> Tuple[List[str], List[str]]:
    top_counter = Counter()
    bot_counter = Counter()
    total = len(pages)
    for p in pages:
        txt = clean_whitespace(p.page_content)
        lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
        if not lines:
            continue
        top_counter.update(lines[:HEADER_FOOTER_TOP_LINES])
        bot_counter.update(lines[-HEADER_FOOTER_BOT_LINES:])
    headers, footers = [], []
    if total == 0:
        return headers, footers
    thr = max(1, int(total * REPEAT_THRESHOLD))
    for line, cnt in top_counter.items():
        if cnt >= thr and len(line) > 2 and not looks_like_page_number(line):
            headers.append(line)
    for line, cnt in bot_counter.items():
        if cnt >= thr and len(line) > 2 and not looks_like_page_number(line):
            footers.append(line)
    return headers, footers

def strip_headers_footers_from_page(text: str, headers: List[str], footers: List[str]) -> str:
    if not text:
        return text
    text = clean_whitespace(text)
    lines = [ln for ln in text.split("\n")]
    # strip top headers
    changed = True
    while changed and headers and lines:
        changed = False
        for h in headers:
            if lines and lines[0].strip() == h.strip():
                lines.pop(0)
                changed = True
    # strip footers
    changed = True
    while changed and footers and lines:
        changed = False
        for f in footers:
            if lines and lines[-1].strip() == f.strip():
                lines.pop(-1)
                changed = True
    # remove page-number like top/bottom lines
    if lines and looks_like_page_number(lines[0]):
        lines.pop(0)
    if lines and looks_like_page_number(lines[-1]):
        lines.pop(-1)
    return "\n".join(lines).strip()

def preprocess_documents(docs: List[Document]) -> List[Document]:
    by_src = defaultdict(list)
    for d in docs:
        src = d.metadata.get("source", "unknown")
        by_src[src].append(d)
    cleaned = []
    for src, pages in by_src.items():
        # treat as paginated if every page has 'page' metadata and len>1
        is_paginated = all(("page" in p.metadata and p.metadata.get("page") is not None) for p in pages) and len(pages) > 1
        if is_paginated:
            headers, footers = detect_repeated_header_footer(pages)
            for p in pages:
                stripped = strip_headers_footers_from_page(p.page_content, headers, footers)
                out_lines = []
                for ln in stripped.split("\n"):
                    ln_s = ln.strip()
                    if looks_like_page_number(ln_s):
                        continue
                    if re.fullmatch(r'[-_=]{3,}', ln_s):
                        continue
                    out_lines.append(ln.rstrip())
                final = clean_whitespace("\n".join(out_lines))
                meta = dict(p.metadata)
                meta["source"] = src
                cleaned.append(Document(page_content=final, metadata=meta))
        else:
            for p in pages:
                txt = clean_whitespace(p.page_content)
                txt = re.sub(r'Page\s*\d+\s*of\s*\d+', '', txt, flags=re.IGNORECASE)
                txt = re.sub(r'\n\s*\d+\s*\n', '\n', txt)
                cleaned.append(Document(page_content=txt, metadata=dict(p.metadata)))
    return cleaned

# splitting using token counts
def split_text_to_token_chunks(text: str, tokenize_fn: Callable, detokenize_fn: Callable,
                               chunk_size: int, chunk_overlap: int):
    tokens = tokenize_fn(text)
    if not tokens:
        return []
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_size must be > chunk_overlap")
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]
        # convert tokens back to text (if token type are ints for tiktoken)
        chunk_text = detokenize_fn(chunk_tokens) if callable(detokenize_fn) else " ".join(map(str, chunk_tokens))
        chunks.append((start, end, chunk_text))
        if end == total:
            break
        start += step
    return chunks

def split_documents_to_chunks(docs: List[Document], tokenize_fn, detokenize_fn, count_tokens_fn,
                              chunk_size=CHUNK_TARGET, chunk_overlap=CHUNK_OVERLAP):
    all_chunks = []
    stats = {"source_counts": defaultdict(int), "total_tokens": 0}
    for doc_idx, d in enumerate(docs):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        text = d.page_content.strip()
        if not text:
            continue
        nt = count_tokens_fn(text)
        stats["total_tokens"] += nt
        stats["source_counts"][source] += 1
        token_chunks = split_text_to_token_chunks(text, tokenize_fn, detokenize_fn, chunk_size, chunk_overlap)
        for ci, (start_idx, end_idx, chunk_text) in enumerate(token_chunks):
            base = os.path.splitext(os.path.basename(source))[0] if isinstance(source, str) else str(source)
            chunk_id = f"{base}_p{page if page is not None else 0}_{doc_idx}_{ci}"
            num_tokens_chunk = count_tokens_fn(chunk_text)
            chunk_meta = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": source,
                "page": page,
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "num_tokens": int(num_tokens_chunk)
            }
            all_chunks.append(chunk_meta)
    return all_chunks, stats

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ingest(data_dir=RAW_DIR, out_dir=OUT_DIR, out_chunks_name=OUT_CHUNKS, out_meta_name=OUT_METADATA,
           chunk_size=CHUNK_TARGET, chunk_overlap=CHUNK_OVERLAP):
    print("Loading raw documents from:", data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    docs: List[Document] = []
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            loaded = loader.load()  # usually returns page-level Documents
            for d in loaded:
                d.metadata.setdefault("source", filepath)
                docs.append(d)
        elif filename.lower().endswith((".txt", ".md")):
            loader = TextLoader(filepath, encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata.setdefault("source", filepath)
                docs.append(d)
        elif filename.lower().endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = None
                    if isinstance(obj, dict):
                        for k in ("text", "content", "body"):
                            if k in obj and isinstance(obj[k], str):
                                text = obj[k]
                                break
                    if text:
                        text = clean_whitespace(text)
                        docs.append(Document(page_content=text, metadata={"source": filepath, "json_line": line_no}))
        elif filename.lower().endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    obj = json.load(f)
                except json.JSONDecodeError:
                    print("Skipping invalid JSON:", filepath)
                    continue
            if isinstance(obj, list):
                for idx, item in enumerate(obj):
                    if isinstance(item, str):
                        text = clean_whitespace(item)
                        docs.append(Document(page_content=text, metadata={"source": filepath, "index": idx}))
                    elif isinstance(item, dict):
                        text = None
                        for k in ("text", "content", "body"):
                            if k in item and isinstance(item[k], str):
                                text = item[k]
                                break
                        if text:
                            text = clean_whitespace(text)
                            docs.append(Document(page_content=text, metadata={"source": filepath, "index": idx}))
            elif isinstance(obj, dict):
                text = None
                for k in ("text", "content", "body"):
                    if k in obj and isinstance(obj[k], str):
                        text = obj[k]
                        break
                if text:
                    text = clean_whitespace(text)
                    docs.append(Document(page_content=text, metadata={"source": filepath}))
                else:
                    dumped = json.dumps(obj)
                    docs.append(Document(page_content=clean_whitespace(dumped), metadata={"source": filepath}))
        else:
            continue

    print(f"Loaded {len(docs)} raw documents/pages.")

    print("Preprocessing documents (header/footer and page-noise removal)...")
    cleaned_docs = preprocess_documents(docs)
    print(f"After preprocessing: {len(cleaned_docs)} documents/pages remain.")

    tokenize_fn, detokenize_fn, count_tokens_fn, tok_name = get_tokenizer()
    print("Tokenizer backend:", tok_name)

    print(f"Splitting into token chunks (target {chunk_size} tokens, overlap {chunk_overlap})...")
    chunks, stats = split_documents_to_chunks(cleaned_docs, tokenize_fn, detokenize_fn, count_tokens_fn,
                                              chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(chunks)} chunks from {len(cleaned_docs)} documents/pages.")

    out_chunks_path = os.path.join(out_dir, out_chunks_name)
    save_json(chunks, out_chunks_path)
    print("Saved chunks to:", out_chunks_path)

    metadata_summary = {
        "total_raw_pages": len(docs),
        "total_cleaned_pages": len(cleaned_docs),
        "total_chunks": len(chunks),
        "tokenizer": tok_name,
        "params": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "source_page_counts": {k: int(v) for k, v in stats["source_counts"].items()},
        "total_tokens": int(stats["total_tokens"])
    }
    out_meta_path = os.path.join(out_dir, out_meta_name)
    save_json(metadata_summary, out_meta_path)
    print("Saved ingestion metadata to:", out_meta_path)
    print("Done.")

if __name__ == "__main__":
    ingest()
