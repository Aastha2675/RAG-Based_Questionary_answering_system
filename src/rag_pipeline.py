# Import Required Libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.base import Embeddings
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

# load environment variables
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# getting the api token
hf_token = os.getenv("HUGGINGFACE_API_KEY")

# initialize re-ranker model
rerank_model = None

def get_reranker():
    global rerank_model
    if rerank_model is None:
        from sentence_transformers import CrossEncoder
        rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return rerank_model



# load PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages


# perform chunking
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


# load vector store
def load_vectorstore(persist_dir="vectorstore"):
    embedding_model = HFEmbedding()

    vectordb = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )

    return vectordb


# retrieve relevant chunks
def retrieve_relevant_chunks(query, vectordb, k=3):
    docs = vectordb.similarity_search(query, k=k)
    return docs


# answer generator
def generate_answer(query, vectordb):
    repo_id = "meta-llama/Llama-3.2-3B-Instruct"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=512,
    )
    chat_model = ChatHuggingFace(llm=llm)

    # re-ranking logic
    # step 0: get the ached model
    rerank_model = get_reranker()

    # step 1: retrieve more chunks than needed 
    initial_docs = vectordb.similarity_search(query, k=10)
    
    # step 2: score them with Cross-Encoder
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = rerank_model.predict(pairs)
    
    # step c: sort and pick top 3
    scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
    final_docs = [doc for score, doc in scored_docs[:3]]

    context = "\n\n".join([doc.page_content for doc in final_docs])

    # extract citations i.e page numbers
    pages = [str(doc.metadata.get('page', 0) + 1) for doc in final_docs]
    citations = ", ".join(sorted(list(set(pages)))) # unique, sorted pages

    messages = [
        SystemMessage(content="You are a financial assistant. Answer strictly based on the provided Swiggy Annual Report context."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
    ]

    response = chat_model.invoke(messages)

    return response.content, context, citations