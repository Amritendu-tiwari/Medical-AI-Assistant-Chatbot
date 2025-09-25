import os
import re
import json
import uuid
import datetime
import streamlit as st
from dotenv import load_dotenv

# Import your QA functions (ensure qa.py exists and defines get_rag_chain & answer_query)
try:
    from qa import get_rag_chain, answer_query
except Exception:
    get_rag_chain = None
    answer_query = None

# Load environment (won't overwrite existing env vars)
load_dotenv("../.env")

# ---------------- session storage ----------------s
SESSIONS_DIR = "data/sessions"
SESSIONS_INDEX = os.path.join(SESSIONS_DIR, "sessions_index.json")
os.makedirs(SESSIONS_DIR, exist_ok=True)

def load_sessions_index():
    if os.path.exists(SESSIONS_INDEX):
        try:
            with open(SESSIONS_INDEX, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sessions_index(index_obj):
    os.makedirs(os.path.dirname(SESSIONS_INDEX), exist_ok=True)
    with open(SESSIONS_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_obj, f, indent=2, ensure_ascii=False)

def create_session_file(name: str):
    session_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    session = {
        "session_id": session_id,
        "name": name,
        "created_at": created_at,
        "messages": []
    }
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    # update index
    idx = load_sessions_index()
    idx.setdefault(name, []).append({"session_id": session_id, "created_at": created_at})
    save_sessions_index(idx)
    return session

def load_session_file(session_id: str):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_session_file(session_obj: dict):
    path = os.path.join(SESSIONS_DIR, f"{session_obj['session_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_obj, f, indent=2, ensure_ascii=False)

def clear_session_messages(session_obj: dict):
    session_obj["messages"] = []
    save_session_file(session_obj)

# ---------------- name extraction helper ----------------
def extract_name(name_input: str) -> str:
    """
    Try to extract a clean name from a free-form first message.
    Removes greetings like "hi", "hello", "my name is", "i am", etc.
    Returns cleaned name or the trimmed input if extraction fails.
    """
    if not name_input:
        return ""
    s = name_input.strip()
    # Remove common lead-ins
    patterns = [
        r'^(hi|hello|hey)[\s,!.-]*',
        r'^(hi, my name is|hello my name is)[\s:,-]*',
        r'^(my name is|i am|i\'m|this is|name is)[\s:,-]*'
    ]
    for p in patterns:
        s = re.sub(p, '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(it's|it is|here|this is)[\s,:-]*", "", s, flags=re.IGNORECASE).strip()
    # Keep up to first two words (first + last)
    words = s.split()
    if len(words) > 2:
        s = " ".join(words[:2])
    # Remove trailing punctuation
    s = s.strip(" ,.!?\"'")
    return s if s else name_input.strip()

# ---------------- Streamlit UI & initialization ----------------
st.set_page_config(page_title="Medical AI Assistant", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")
st.title("🏥 Medical AI Assistant")

# session_state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "session_obj" not in st.session_state:
    st.session_state.session_obj = None
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = get_rag_chain() if get_rag_chain is not None else None
    except Exception as e:
        st.session_state.rag_chain = None
        st.session_state.init_error = str(e)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🏥 Medical AI Assistant")
    st.markdown("---")
    st.markdown(
        """
        **About**  
        This is a medical AI assistant that provides **general medical information** based on trusted sources.
        
        **Important disclaimer**  
        ⚠️ This assistant provides educational information only — it does **not** provide diagnoses, prescriptions, or personalized medical advice.
        Always consult a qualified healthcare professional for medical concerns.
        """
    )
    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        if st.session_state.session_obj is not None:
            clear_session_messages(st.session_state.session_obj)
            st.session_state.messages = []
            st.success("Chat history cleared for current session.")
            st.rerun()
        else:
            st.session_state.messages = []
            st.info("Cleared in-memory chat history.")
            st.rerun()

# ---------------- First-time name flow ----------------
if st.session_state.user_name is None:
    # assistant asks for name
    with st.chat_message("assistant"):
        st.markdown("Hello! Before we begin, may I know your name?")

    if name_raw := st.chat_input("Type your name here..."):
        cleaned_name = extract_name(name_raw)
        if not cleaned_name:
            st.warning("Please enter a valid name (e.g., 'Amritendu').")
            st.stop()
        # create session and greeting
        session_obj = create_session_file(cleaned_name)
        st.session_state.user_name = cleaned_name
        st.session_state.session_obj = session_obj

        # save standardized user entry, not the full raw utterance
        user_msg = {"role": "user", "content": f"Name: {cleaned_name}", "ts": datetime.datetime.utcnow().isoformat() + "Z"}
        assistant_msg = {"role": "assistant", "content": f"Nice to meet you, {cleaned_name}! How can I help you with general medical information today?", "ts": datetime.datetime.utcnow().isoformat() + "Z"}

        session_obj["messages"].append(user_msg)
        session_obj["messages"].append(assistant_msg)
        save_session_file(session_obj)

        st.session_state.messages = [user_msg, assistant_msg]
        st.rerun()
    st.stop()

# ---------------- Main chat UI ----------------
# Ensure session object exists
if st.session_state.session_obj is None:
    st.session_state.session_obj = create_session_file(st.session_state.user_name)

session = st.session_state.session_obj
st.caption(f"**Session ID:** {session['session_id']} — **User:** {session['name']} — Created: {session['created_at']}")

# Prefer persistent session messages, fallback to session_state.messages buffer
display_messages = session.get("messages", []) if session.get("messages") else st.session_state.messages

for m in display_messages:
    role = m.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(m.get("content", ""))
        if m.get("sources"):
            with st.expander("View Sources"):
                for i, s in enumerate(m["sources"], start=1):
                    src = s.get("source", "unknown")
                    page = s.get("page", None)
                    preview = s.get("preview") or s.get("content", "")
                    st.markdown(f"**Source {i}:** {src}  \n**Page:** {page}")
                    st.markdown(f"> {preview[:400]}")

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # append user message to session & save
    user_msg = {"role": "user", "content": prompt, "ts": datetime.datetime.utcnow().isoformat() + "Z"}
    session["messages"].append(user_msg)
    save_session_file(session)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.rag_chain is None:
            err = st.session_state.get("init_error", "RAG chain not initialized.")
            st.error(f"RAG chain unavailable: {err}")
            reply = "I'm unable to answer right now. Please try again later or consult a healthcare professional."
            st.markdown(reply)
            assistant_msg = {"role": "assistant", "content": reply, "sources": [], "ts": datetime.datetime.utcnow().isoformat() + "Z"}
            session["messages"].append(assistant_msg)
            save_session_file(session)
            st.session_state.messages.append(assistant_msg)
        else:
            try:
                with st.spinner("Thinking..."):
                    result = answer_query(st.session_state.rag_chain, prompt)
                    answer = result.get("answer") if isinstance(result, dict) else str(result)
                    sources = result.get("sources", []) if isinstance(result, dict) else []

                    st.markdown(answer)

                    assistant_msg = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "ts": datetime.datetime.utcnow().isoformat() + "Z"
                    }
                    session["messages"].append(assistant_msg)
                    save_session_file(session)
                    st.session_state.messages.append(assistant_msg)

                    if sources:
                        with st.expander("View Sources"):
                            for i, s in enumerate(sources, start=1):
                                src = s.get("source", "unknown")
                                page = s.get("page", None)
                                preview = s.get("preview") or s.get("content", "")
                                st.markdown(f"**Source {i}:** {src}  \n**Page:** {page}")
                                st.markdown(f"> {preview[:400]}")
            except Exception as e:
                err_msg = f"Sorry — an error occurred while processing your request: {e}"
                st.error(err_msg)
                assistant_msg = {"role": "assistant", "content": "Sorry — an internal error occurred.", "sources": [], "ts": datetime.datetime.utcnow().isoformat() + "Z"}
                session["messages"].append(assistant_msg)
                save_session_file(session)
                st.session_state.messages.append(assistant_msg)

# Footer disclaimer (always visible, not saved in messages)
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.85em;'>"
    "<strong>Important Medical Disclaimer:</strong><br>"
    "This AI assistant provides general medical information only and does not provide medical advice, diagnoses, or treatment recommendations. "
    "Always consult a qualified healthcare professional for medical concerns."
    "</div>",
    unsafe_allow_html=True
)
