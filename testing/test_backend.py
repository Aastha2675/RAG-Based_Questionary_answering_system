import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import *

pdf_path = "data/Swiggy Annual Report.pdf"

pages = load_pdf(pdf_path)
print("Pages loaded:", len(pages))

chunks = chunk_pages(pages)
print("Chunks created:", len(chunks))

vectordb = create_vectorstore(chunks)
print("Vectorstore created!")