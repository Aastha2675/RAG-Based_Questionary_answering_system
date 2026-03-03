import os
import streamlit as st
from rag_pipeline import load_vectorstore, generate_answer

# ---------------------
# Path Configuration
# ---------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(ROOT_DIR, "vectorstore")

# ---------------------
# Load Vector DB
# ---------------------
@st.cache_resource
def load_db():
    return load_vectorstore(VECTORSTORE_PATH)

try:
    vectordb = load_db()
except Exception as e:
    st.error(f"Error: Vectorstore not found at {VECTORSTORE_PATH}. Check your folder structure!")
    st.stop()

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Swiggy RAG", page_icon="🍔")
st.title("Swiggy Annual Report – QA System")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing the report..."):
            try:
                # Direct call to generate_answer
                answer, context = generate_answer(query, vectordb)

                st.subheader("Answer:")
                st.success(answer)

                with st.expander("Show Context Chunks"):
                    st.write(context)
            except Exception as e:
                st.error(f"Error calling LLM API: {e}")