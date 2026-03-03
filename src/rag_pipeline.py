# import all the required libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
import os
import streamlit as st

# load HF API key from streamlit secrets
hf_token = st.secrets["HUGGINGFACE_API_KEY"]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load pdf and convert text
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# perform Chunking
def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(pages)

# embedding model
class HFEmbedding(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# create vector store
def create_vectorstore(chunks, persist_dir="vectorstore"):
    embedding_model = HFEmbedding()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

# load vector store (for query)
def load_vectorstore(persist_dir="vectorstore"):
    return Chroma(
        embedding_function=HFEmbedding(),
        persist_directory=persist_dir
    )

# retriever
def retrieve_relevant_chunks(query, vectordb, k=3):
    return vectordb.similarity_search(query, k=k)

# generate answer
def generate_answer(query, vectordb):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
        temperature=0.2,
        max_new_tokens=512,
    )

    chat_model = ChatHuggingFace(llm=llm)

    docs = retrieve_relevant_chunks(query, vectordb)
    context = "\n\n".join([d.page_content for d in docs])

    messages = [
        SystemMessage(content="You are a financial assistant. Answer strictly based on the Swiggy Annual Report."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
    ]

    response = chat_model.invoke(messages)
    return response.content, context