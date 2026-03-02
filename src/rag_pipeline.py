# import all the required libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.base import Embeddings
import os

# load pdf and convert text
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()   
    return pages

# perform Chuncking
def chunk_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


# embedding model 
class HFEmbedding(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]  

    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()  
    
    
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
    embedding_model = HFEmbedding()
    vectordb = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    return vectordb

# retriever (query function)
def retrieve_relevant_chunks(query, vectordb, k=3):
    docs = vectordb.similarity_search(query, k=k)
    return docs

# answer generator
def generate_answer(query, vectordb, llm):
    retrieved_docs = retrieve_relevant_chunks(query, vectordb)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""
    You are an AI assistant that answers ONLY using the Swiggy Annual Report.
    
    Context:
    {context}

    Question:
    {query}

    Answer strictly from the context (no outside knowledge):
    """

    answer = llm(prompt)
    return answer, context