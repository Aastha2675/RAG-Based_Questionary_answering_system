import os
import streamlit as st
from rag_pipeline import load_vectorstore, generate_answer

# Page Config
st.set_page_config(
    page_title="Swiggy Intelligence Bot",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
header { visibility: hidden; }

.stApp {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    max-width: 800px;
}

.header-text {
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #FFFFFF;
    margin-bottom: 0px;
    font-weight: 800;
}

.sub-text {
    color: #A0AEC0;
    margin-top: -10px;
}

[data-testid="stChatMessage"] {
    background-color: #1A202C;
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 10px;
    border: 1px solid #2D3748;
    color: #FFFFFF;
}

.stChatInputContainer textarea {
    background-color: #2D3748 !important;
    color: white !important;
    border-color: #4A5568 !important;
}

.stButton>button {
    background-color: #FC8019;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #e67316;
    color: white;
}

hr {
    border-top: 1px solid #2D3748 !important;
}

[data-testid="column"] {
    display: flex;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# Path Config
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(ROOT_DIR, "vectorstore")
logo_path = os.path.join(ROOT_DIR, "assets", "swiggy_logo.jpeg")

# Header Section
header_col1, header_col2 = st.columns([0.18, 0.82])

with header_col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=70)
    else:
        st.write("🍔")

with header_col2:
    st.markdown(
        "<h1 class='header-text'>Swiggy Annual Report AI Assistant</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p class='sub-text'>Annual Report 2023-24 Intelligence System</p>",
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# Chat History Storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Vector DB
@st.cache_resource
def load_db():
    if not os.path.exists(VECTORSTORE_PATH):
        st.error("Vectorstore not found! Please run test_backend.py first.")
        return None
    return load_vectorstore(VECTORSTORE_PATH)

vectordb = load_db()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
query = st.chat_input("Ask a question about Swiggy's Annual Report...")

if query:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Assistant Response
    if vectordb:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing report..."):
                try:
                    answer, context = generate_answer(query, vectordb)

                    st.markdown(answer)

                    with st.expander("Source Context"):
                        st.caption(context)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Vectorstore is not initialized.")

# Clear Chat Button
if len(st.session_state.messages) > 0:
    st.write("")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()