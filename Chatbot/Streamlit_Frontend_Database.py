import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_db_backend import (
    chatbot,
    load_titles,
    make_title,
    retrieve_all_threads,
    save_title,
)

st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", layout="centered")

NEW_CHAT_LABEL = "New chat"

# ***************************** Styling (pure CSS: no runtime cost) *****************************
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.5rem; max-width: 780px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #f7f7f8; border-right: 1px solid #ececf1; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; text-align: left; justify-content: flex-start;
    border: none; background: transparent; color: #343541;
    border-radius: 10px; padding: .55rem .75rem; font-size: .9rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: background .15s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #ececf1; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #e4e4ea; font-weight: 600; color: #111;
}
.side-title { font-size: 1.15rem; font-weight: 600; margin-bottom: .75rem; }
.side-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
              color: #8e8ea0; margin: 1.1rem 0 .3rem .4rem; }

/* Chat bubbles */
[data-testid="stChatMessage"] { border-radius: 14px; padding: .8rem 1rem; margin-bottom: .5rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #f1f3f9; }

/* Empty state */
.hero { text-align: center; margin-top: 22vh; }
.hero h1 { font-size: 2rem; font-weight: 600; margin-bottom: .3rem; }
.hero p  { color: #8e8ea0; }

@media (prefers-color-scheme: dark) {
    section[data-testid="stSidebar"] { background: #171717; border-color: #2a2a2a; }
    section[data-testid="stSidebar"] .stButton > button { color: #ececec; }
    section[data-testid="stSidebar"] .stButton > button:hover { background: #262626; }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #2f2f2f; color: #fff; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #262730; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ***************************** Utility functions *****************************
def generate_thread_id():
    # Plain string: the DB stores/returns thread ids as strings, so types always match
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Register a (new) thread at the top of the list. Does NOT re-read the DB."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)
    st.session_state["chat_titles"].setdefault(thread_id, NEW_CHAT_LABEL)


def reset_chat():
    # Already on an empty chat? Reuse it instead of piling up blank "New chat" entries.
    if not st.session_state["message_history"]:
        return
    st.session_state["thread_id"] = generate_thread_id()
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def select_thread(thread_id):
    """Runs as a button callback, so the switch happens before the page re-renders."""
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in load_conversation(thread_id)
    ]


# ***************************** Session setup *****************************
if "chat_threads" not in st.session_state:
    # One-time load of saved chats (newest first) and their labels from SQLite
    threads = retrieve_all_threads()
    st.session_state["chat_threads"] = threads
    st.session_state["chat_titles"] = load_titles(threads)

st.session_state.setdefault("message_history", [])
st.session_state.setdefault("thread_id", generate_thread_id())
add_thread(st.session_state["thread_id"])

# Read input first (the chat box stays pinned at the bottom regardless) so a new
# chat's label can appear in the sidebar on the very same run.
user_input = st.chat_input("Message LangGraph Chatbot…")

if user_input and st.session_state["chat_titles"].get(st.session_state["thread_id"]) == NEW_CHAT_LABEL:
    title = make_title(user_input)
    st.session_state["chat_titles"][st.session_state["thread_id"]] = title
    save_title(st.session_state["thread_id"], title)


# ***************************** Sidebar *****************************
with st.sidebar:
    st.markdown('<div class="side-title">💬 LangGraph Chatbot</div>', unsafe_allow_html=True)
    st.button("＋  New chat", on_click=reset_chat, use_container_width=True)
    st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)

    for tid in st.session_state["chat_threads"]:  # already newest first
        st.button(
            st.session_state["chat_titles"].get(tid, NEW_CHAT_LABEL),
            key=f"thread_{tid}",
            on_click=select_thread,
            args=(tid,),
            type="primary" if tid == st.session_state["thread_id"] else "secondary",
            use_container_width=True,
        )


# ***************************** Main UI *****************************
if not st.session_state["message_history"] and not user_input:
    st.markdown(
        '<div class="hero"><h1>How can I help you today?</h1>'
        "<p>Start a conversation below.</p></div>",
        unsafe_allow_html=True,
    )

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})