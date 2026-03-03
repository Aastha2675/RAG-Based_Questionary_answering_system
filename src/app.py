import os
import streamlit as st
from rag_pipeline import load_vectorstore, generate_answer

# page config
st.set_page_config(
    page_title="Swiggy Intelligence Bot",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="collapsed" 
)

# css style
st.markdown("""
    <style>
    /* Hide the sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #FC8019;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    /* Swiggy Orange focus for input */
    .stChatInputContainer textarea {
        border-color: #FC8019 !important;
    }
    </style>
    """, unsafe_allow_html=True) 

# path config
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(ROOT_DIR, "vectorstore")

# header section
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/12/Swiggy_logo.svg/1200px-Swiggy_logo.svg.png", width=70)
with col2:
    st.title("Swiggy Annual Report AI Assistant")

st.markdown("---") 

# for chat history storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# load the vectordb
@st.cache_resource
def load_db():
    if not os.path.exists(VECTORSTORE_PATH):
        st.error("Vectorstore not found! Please run test_backend.py first.")
        return None
    return load_vectorstore(VECTORSTORE_PATH)

vectordb = load_db()

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# clear chat history
if len(st.session_state.messages) > 0:
    st.write("") # Spacer
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# chat input
if query := st.chat_input("Ask a question about Swiggy's Annual Report..."):
    # user msg query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # assistant response
    if vectordb:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing report..."):
                try:
                    answer, context = generate_answer(query, vectordb)
                    st.markdown(answer)
                    
                    # view context
                    with st.expander("Source Context"):
                        st.caption(context)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Vectorstore is not initialized.")