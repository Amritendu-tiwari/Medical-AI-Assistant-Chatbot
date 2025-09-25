# qa.py
import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        ChatOpenAI = None

try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except Exception:
    RetrievalQA = None
    PromptTemplate = None

try:
    from retriever import get_retriever
except Exception:
    get_retriever = None

# Load environment
load_dotenv("../.env")

# ---------- Config ----------
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 5

DISCLAIMER = (
    "Disclaimer: This information is for general knowledge and informational purposes only, "
    "and does not constitute medical advice. Please consult a qualified healthcare professional for any medical concerns."
)

RED_FLAG_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath", "unconscious",
    "severe bleeding", "heavy bleeding", "not breathing", "loss of consciousness",
    "sudden weakness", "sudden numbness", "slurred speech", "severe abdominal pain",
    "high fever and seizures", "severe head injury", "severe burn", "severe allergic reaction",
    "blue lips", "rapid pulse", "severe trauma"
]

# ---------- Sanitizer ----------
def _sanitize_response(text: str) -> str:
    """Remove unwanted boilerplate phrases and tidy whitespace."""
    if not isinstance(text, str):
        return text
    bad_phrases = [
        "Based on the provided context",
        "Based on the context provided",
        "According to the provided context",
        "According to the context"
    ]
    for phrase in bad_phrases:
        text = text.replace(phrase, "")
    text = re.sub(r"\s{2,}", " ", text).strip()
    if DISCLAIMER not in text:
        text += "\n\n" + DISCLAIMER
    return text

# ---------- Prompt ----------
SYSTEM_INSTRUCTIONS = (
    "You are a careful medical information assistant. Follow these rules exactly:\n\n"
    "1) Use ONLY the provided context. If the context doesn't contain the needed info, say you cannot answer.\n"
    "2) NEVER provide diagnoses, prescriptions, or personalized treatment.\n"
    "3) You MAY provide general, non-prescriptive self-care suggestions (rest, fluids, when to seek care).\n"
    f"4) Always append this disclaimer:\n\"{DISCLAIMER}\"\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n"
)

PROMPT = PromptTemplate(
    template=SYSTEM_INSTRUCTIONS,
    input_variables=["context", "question"]
) if PromptTemplate else None

# ---------- Helpers ----------
def _detect_red_flag(query: str) -> List[str]:
    q = (query or "").lower()
    return [kw for kw in RED_FLAG_KEYWORDS if kw in q]

# ---------- Public API ----------
def get_rag_chain(model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, top_k: int = DEFAULT_TOP_K):
    if get_retriever is None:
        raise RuntimeError("Retriever not available. Ensure retriever.py defines get_retriever().")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Put it in ../.env or export it.")

    retriever = get_retriever(top_k=top_k)

    if ChatOpenAI is None:
        raise ImportError("ChatOpenAI not found. Install/upgrade langchain_openai.")
    try:
        llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)
    except TypeError:
        llm = ChatOpenAI(model=model_name, temperature=temperature)

    if RetrievalQA is None:
        raise ImportError("RetrievalQA not available.")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT} if PROMPT else {}
    )
    return chain

def answer_query(chain, query: str) -> Dict[str, Any]:
    if not chain:
        raise ValueError("No chain provided")

    red_matches = _detect_red_flag(query)
    if red_matches:
        urgent_msg = (
            "⚠️ Potential emergency symptoms detected: "
            + ", ".join(red_matches[:3])
            + ". If you or someone has these symptoms, seek emergency care immediately or call your local emergency number."
        )
        urgent_msg = _sanitize_response(urgent_msg)
        return {"answer": urgent_msg, "sources": [], "raw": {"red_flags": red_matches}}

    try:
        out = chain({"query": query})
    except TypeError:
        out = chain({"input": query})

    answer_text = None
    if isinstance(out, dict):
        answer_text = out.get("result") or out.get("answer") or out.get("output_text")
        source_docs = out.get("source_documents") or []
    else:
        answer_text = str(out)
        source_docs = []

    if not answer_text:
        answer_text = str(out)

    answer_text = _sanitize_response(answer_text)

    sources = []
    for d in source_docs:
        try:
            meta = getattr(d, "metadata", {}) or {}
            content = getattr(d, "page_content", "")
            preview = content[:400] + ("..." if len(content) > 400 else "")
            sources.append({
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "preview": preview
            })
        except Exception:
            sources.append({"preview": str(d)[:400]})

    return {"answer": answer_text, "sources": sources, "raw": out}

# ---------- Run standalone ----------
if __name__ == "__main__":
    chain = get_rag_chain()
    q = "Tell me about vitamin D"
    res = answer_query(chain, q)
    print(res["answer"])
