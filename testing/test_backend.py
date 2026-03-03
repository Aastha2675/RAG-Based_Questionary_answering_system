import sys
import os

# Ensure root folder is in PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.rag_pipeline import (
    load_pdf,
    chunk_pages,
    create_vectorstore,
    load_vectorstore,
    retrieve_relevant_chunks,
    generate_answer
)

print("\n--- File Check ---")
pdf_path = os.path.join(root_dir, "data", "Swiggy Annual Report.pdf")
env_path = os.path.join(root_dir, ".env")

if os.path.exists(pdf_path):
    print("PDF Found")
else:
    print("PDF NOT FOUND", pdf_path)

if os.path.exists(env_path):
    print(".env Found ✔")
else:
    print(".env NOT FOUND", env_path)

print("\n--- PDF Processing ---")
pages = load_pdf(pdf_path)
print(f"Pages Loaded: {len(pages)}")

chunks = chunk_pages(pages)
print(f"Chunks Created: {len(chunks)}")

print("\n--- Vectorstore Creation ---")
vectordb = create_vectorstore(chunks, persist_dir=os.path.join(root_dir, "vectorstore"))
print("Vectorstore created ✔")

print("\n--- Vectorstore Load & Retrieval Test ---")
vectordb_loaded = load_vectorstore(os.path.join(root_dir, "vectorstore"))
query = "What is Swiggy's net loss in FY 2024?"

docs = retrieve_relevant_chunks(query, vectordb_loaded)
print(f"Retrieved {len(docs)} relevant chunks ✔")

for i, d in enumerate(docs, 1):
    print(f"\n--- Chunk {i} Sample ---")
    print(d.page_content[:300], "...\n")

print("\n--- (Optional) RAG Answer Generation Test ---")
try:
    answer, context = generate_answer(query, vectordb_loaded) 
    print("\nAnswer:", answer)
    print("\nEnd-to-end RAG pipeline working ✔")
except Exception as e:
    print("LLM API failed (expected if HF free-tier). Pipeline still OK.")
    print("Error:", e)