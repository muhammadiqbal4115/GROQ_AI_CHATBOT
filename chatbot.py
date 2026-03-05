# Import
import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


# Load env
load_dotenv()
ENV_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


# Streamlit page config
st.set_page_config(
    page_title="Groq Chatbot (With Temperary Memory)",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Conversational AI Chatbot")
st.caption("Built with Streamlit + Langchain + Groq Cloud API")

# Default system prompt
default_prompt = "You are a helpful AI Assistant. Be clear, correct and concise"


# Sidebar
with st.sidebar:

    st.header("⚙️ Controls")

    api_key_input = st.text_input(
        "Groq API Key (optional)",
        type="password"
    )

    GROQ_API_KEY = api_key_input.strip() if api_key_input.strip() else ENV_GROQ_API_KEY


    model_name = st.selectbox(
        "Choose Model",
        [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "whisper-large-v3-turbo",
            "groq/compounds"
        ],
        index=0
    )


    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1
    )


    max_tokens = st.slider(
        "Max Tokens (Reply length)",
        min_value=64,
        max_value=2048,
        value=640,
        step=64
    )

    # Tone preset
    tone_preset = st.selectbox(
        "Tone Preset",
        ["Custom", "Friendly", "Strict", "Teacher"]
    )

    tone_prompts = {
        "Friendly": "You are a friendly AI assistant. Respond warmly and politely.",
        "Strict": "You are a strict and professional AI assistant. Give short and precise answers.",
        "Teacher": "You are a teacher. Explain concepts clearly with examples so the user can learn."
    }

    if tone_preset != "Custom":
        system_prompt = tone_prompts[tone_preset]

    # System Prompt
    system_prompt = st.text_area(
        "System Prompt (Rules for the bot)",
        value=default_prompt,
        height=140
    )


    # Reset system prompt
    if st.button("Reset System Prompt"):
        system_prompt = default_prompt
        st.rerun()


    typing_effect = st.checkbox("Enable typing effect", value=True)

    st.divider()


    # Clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.pop("history_store", None)
        st.rerun()


# API Key Guard
if not GROQ_API_KEY:
    st.error("🔑 Groq API Key is missing. Add it in your .env or paste it in the sidebar")
    st.stop()


# Chat history store
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

SESSION_ID = "default_session"


def get_history(session_id: str) -> InMemoryChatMessageHistory:

    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.history_store[session_id]


# Build LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)


# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


chain = prompt | llm | StrOutputParser()


# Add memory
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)


# Render old messages
history_obj = get_history(SESSION_ID)

for msg in history_obj.messages:

    role = getattr(msg, "type", "")

    if role == "human":
        st.chat_message("user").write(msg.content)

    else:
        st.chat_message("assistant").write(msg.content)


# User input
user_input = st.chat_input("Type your message....")


if user_input:

    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        try:

            response_text = chat_with_history.invoke(
                {"input": user_input, "system_prompt": system_prompt},
                config={"configurable": {"session_id": SESSION_ID}},
            )

        except Exception as e:

            st.error(f"Model Error: {e}")
            response_text = ""


        # Typing effect
        if typing_effect and response_text:

            typed = ""

            for ch in response_text:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.005)

        else:
            placeholder.write(response_text)


# Export section
st.divider()

st.subheader("⬇️ Download Chat History")


# JSON export
export_data = []

for m in get_history(SESSION_ID).messages:

    role = getattr(m, "type", "")

    if role == "human":
        export_data.append({"role": "user", "text": m.content})

    else:
        export_data.append({"role": "assistant", "text": m.content})


st.download_button(
    label="Download chat history in json",
    data=json.dumps(export_data, ensure_ascii=False, indent=2),
    file_name="chat_history.json",
    mime="application/json",
)


# TXT export
txt_data = ""

for m in get_history(SESSION_ID).messages:

    role = getattr(m, "type", "")

    if role == "human":
        txt_data += f"User: {m.content}\n\n"

    else:
        txt_data += f"Assistant: {m.content}\n\n"


st.download_button(
    label="Download chat history in txt",
    data=txt_data,
    file_name="chat_history.txt",
    mime="text/plain",
)