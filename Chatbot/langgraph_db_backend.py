import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()  # ChatGroq reads GROQ_API_KEY from the environment itself

# Model
llm = ChatGroq(model="openai/gpt-oss-120b")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)
checkpointer.setup()  # make sure the tables exist before we query them

# Titles live in the same DB so they survive restarts
with checkpointer.lock:
    conn.execute("CREATE TABLE IF NOT EXISTS chat_titles (thread_id TEXT PRIMARY KEY, title TEXT NOT NULL)")
    conn.commit()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# helpers
def retrieve_all_threads():
    """All thread ids that have saved checkpoints, newest first (fast direct SQL)."""
    with checkpointer.lock:
        rows = conn.execute(
            "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
        ).fetchall()
    return [r[0] for r in rows]


def make_title(text, limit=32):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def save_title(thread_id, title):
    with checkpointer.lock:
        conn.execute(
            "INSERT INTO chat_titles (thread_id, title) VALUES (?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title",
            (str(thread_id), title),
        )
        conn.commit()


def load_titles(thread_ids):
    """Return {thread_id: title}. Threads without a saved title (e.g. created before
    titles existed) get one from their first user message, saved once."""
    with checkpointer.lock:
        titles = dict(conn.execute("SELECT thread_id, title FROM chat_titles").fetchall())

    for tid in thread_ids:
        if tid in titles:
            continue
        state = chatbot.get_state(config={"configurable": {"thread_id": tid}})
        msgs = state.values.get("messages", [])
        first = next((m.content for m in msgs if m.type == "human"), None)
        if first:
            titles[tid] = make_title(first)
            save_title(tid, titles[tid])
    return titles