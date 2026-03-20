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
    page_title="Groq Chatbot (With Temporary Memory)",
    page_icon="🤖",
    layout="centered"
)


# ── NEON PURPLE THEME ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --purple-deep:    #0a0010;
    --purple-dark:    #110022;
    --purple-mid:     #1e003a;
    --purple-glass:   rgba(140, 0, 255, 0.08);
    --neon-core:      #bf00ff;
    --neon-bright:    #d966ff;
    --neon-soft:      #9b30ff;
    --neon-dim:       #6a0aad;
    --neon-glow:      0 0 8px #bf00ff, 0 0 20px #9b30ff55, 0 0 40px #7b00ff33;
    --neon-glow-text: 0 0 6px #bf00ff, 0 0 18px #9b30ff;
    --border-glow:    1px solid rgba(191, 0, 255, 0.4);
    --text-primary:   #f0d6ff;
    --text-secondary: #b38fd9;
    --text-muted:     #7a5a9a;
    --font-display:   'Orbitron', monospace;
    --font-body:      'Rajdhani', sans-serif;
}

/* ── Global reset & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--purple-deep) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* Animated grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(191, 0, 255, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(191, 0, 255, 0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] {
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--purple-dark); }
::-webkit-scrollbar-thumb {
    background: var(--neon-core);
    border-radius: 2px;
    box-shadow: var(--neon-glow);
}

/* ── Title / Header ── */
[data-testid="stMarkdownContainer"] h1,
.stTitle, h1 {
    font-family: var(--font-display) !important;
    font-weight: 900 !important;
    font-size: 1.9rem !important;
    background: linear-gradient(135deg, #fff 0%, #d966ff 40%, #bf00ff 70%, #7b00ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    letter-spacing: 0.06em;
    margin-bottom: 0 !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--purple-dark) !important;
    border-right: var(--border-glow) !important;
    box-shadow: 4px 0 30px rgba(140, 0, 255, 0.15);
}

[data-testid="stSidebar"] * {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    color: var(--neon-bright) !important;
    text-shadow: var(--neon-glow-text);
    text-transform: uppercase;
}

/* ── Inputs & Selects ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] {
    background: var(--purple-glass) !important;
    border: var(--border-glow) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    transition: box-shadow 0.2s, border-color 0.2s;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    box-shadow: var(--neon-glow) !important;
    border-color: var(--neon-bright) !important;
    outline: none !important;
}

/* Password input eye icon */
[data-testid="stTextInput"] button {
    color: var(--neon-bright) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--neon-core) !important;
    box-shadow: var(--neon-glow) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"]::before {
    background: var(--neon-dim) !important;
}

[data-testid="stSlider"] div[style*="background"] {
    background: linear-gradient(90deg, var(--neon-dim), var(--neon-core)) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    font-family: var(--font-display) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--neon-bright) !important;
    border: 1px solid var(--neon-core) !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 10px rgba(191, 0, 255, 0.2), inset 0 0 10px rgba(191, 0, 255, 0.05) !important;
    position: relative;
    overflow: hidden;
}

[data-testid="stButton"] > button:hover {
    background: rgba(191, 0, 255, 0.15) !important;
    box-shadow: 0 0 20px rgba(191, 0, 255, 0.5), 0 0 40px rgba(155, 48, 255, 0.3), inset 0 0 15px rgba(191, 0, 255, 0.1) !important;
    color: #fff !important;
    transform: translateY(-1px);
}

[data-testid="stButton"] > button:active {
    transform: translateY(0px);
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    font-family: var(--font-display) !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, rgba(191,0,255,0.2), rgba(123,0,255,0.2)) !important;
    color: var(--neon-bright) !important;
    border: 1px solid var(--neon-core) !important;
    border-radius: 6px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 12px rgba(191, 0, 255, 0.25) !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, rgba(191,0,255,0.4), rgba(123,0,255,0.4)) !important;
    box-shadow: 0 0 25px rgba(191, 0, 255, 0.55) !important;
    transform: translateY(-1px);
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label span {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stCheckbox"] [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--purple-glass) !important;
    border: var(--border-glow) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
    backdrop-filter: blur(10px) !important;
    transition: box-shadow 0.2s;
    font-family: var(--font-body) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 0 18px rgba(191, 0, 255, 0.25) !important;
}

/* User message — slightly brighter tint */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(191, 0, 255, 0.1) !important;
    border-color: rgba(191, 0, 255, 0.5) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(30, 0, 58, 0.9) !important;
    border: 1px solid rgba(191, 0, 255, 0.5) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 1rem !important;
    transition: box-shadow 0.2s, border-color 0.2s;
}

[data-testid="stChatInput"] textarea:focus {
    box-shadow: var(--neon-glow) !important;
    border-color: var(--neon-bright) !important;
}

[data-testid="stChatInput"] button {
    background: var(--neon-core) !important;
    border-radius: 8px !important;
    transition: box-shadow 0.2s !important;
}

[data-testid="stChatInput"] button:hover {
    box-shadow: var(--neon-glow) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--neon-core), transparent) !important;
    margin: 1.5rem 0 !important;
    box-shadow: 0 0 8px rgba(191, 0, 255, 0.5) !important;
}

/* ── Section subheaders ── */
[data-testid="stMarkdownContainer"] h2,
h2, .stSubheader {
    font-family: var(--font-display) !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--neon-bright) !important;
    text-shadow: var(--neon-glow-text) !important;
}

/* ── Error / info boxes ── */
[data-testid="stAlert"] {
    background: rgba(191, 0, 255, 0.07) !important;
    border: 1px solid var(--neon-dim) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
}

/* ── Select dropdown menu ── */
[data-baseweb="popover"] {
    background: var(--purple-mid) !important;
    border: var(--border-glow) !important;
    box-shadow: 0 8px 40px rgba(140, 0, 255, 0.4) !important;
}

[data-baseweb="menu"] li {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

[data-baseweb="menu"] li:hover {
    background: rgba(191, 0, 255, 0.15) !important;
    color: #fff !important;
}

/* ── Tooltip / labels ── */
[data-testid="stWidgetLabel"] p,
label p {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
}

/* ── Pulse animation for the title icon ── */
@keyframes neon-pulse {
    0%, 100% { opacity: 1; filter: drop-shadow(0 0 6px #bf00ff); }
    50%       { opacity: 0.75; filter: drop-shadow(0 0 18px #bf00ff); }
}

/* ── Scanline overlay ── */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 0, 0, 0.07) 2px,
        rgba(0, 0, 0, 0.07) 4px
    );
    pointer-events: none;
    z-index: 1;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────


st.title("⚡ Conversational AI Chatbot")
st.caption("Built with Streamlit · LangChain · Groq Cloud API")

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
    st.error("🔑 Groq API Key is missing. Add it in your .env or paste it in the sidebar.")
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
    label="Download chat history in JSON",
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
    label="Download chat history in TXT",
    data=txt_data,
    file_name="chat_history.txt",
    mime="text/plain",
)
