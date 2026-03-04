# Swiggy Intelligence Bot 
### AI-Powered Annual Report Analysis System

This repository contains a **Retrieval-Augmented Generation (RAG)** application designed to analyze and answer queries based on the **Swiggy Annual Report 2023-24**. Developed as a technical assignment for the **Newel Technology** ML Intern recruitment.

---

## 🎯 Project Objective
An advanced AI system built to accurately answer user questions based strictly on the Swiggy Annual Report 2023-24. This project implements a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed to prevent hallucinations and provide context-grounded financial insights.

The objective of this system is to allow users to ask natural language questions regarding Swiggy's business operations and receive accurate answers derived solely from the official document.

![App Interface](assets/app_screenshot.png)


## 🛠️ Tech Stack
* **LLM:** Llama-3.2-3B-Instruct (via Hugging Face)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **UI Framework:** Streamlit
* **Document Parsing:** PyPDF with Recursive Character Splitting

---

## System Architecture 
![Swiggy Intelligence Bot Architecture](assets/architecture_diagram.png)

## 🚀 Key Features
- **Context-Strict QA:** A specialized "No-Hallucination" engine that answers queries using only the provided annual report.
- **Semantic Retrieval:** Leverages vector embeddings to understand the intent behind financial terms
- **Source Verification**: Includes an expandable Source Context feature for users to audit the exact report snippets used for each answer.
- **Persistent Knowledge Base:** Local vector storage via ChromaDB ensures fast, one-time document processing.
- **Intuitive UI:** Brand-aligned Streamlit interface optimized for financial data exploration.
- **Multi-Query Retrieval:** Implemented to handle non search-friendly user queries by generating multiple query variations to improve retrieval accuracy
- **Cross-Encoder Re-ranking:** Implemented to filter less relevant retrieved chunks and select the most relevant context for answer generation.

---

## 📂 Project Structure
```text
Swiggy_Intelligence_Bot/
├── assets/             # Project images 
├── data/               # Source PDF (Swiggy Annual Report)
├── model_evaluation/        # Automated Ragas performance reports (CSV)
├── src/
│   ├── rag_pipeline.py # Core RAG logic (Chunking, Embedding, Generation)
│   └── app.py          # Streamlit Interactive UI
|   └── evaluation_rag.py    # Automated Evaluation Harness
├── testing/
|   ├── test_backend.py # Backend validation 
├── .env                # API Keys
└── requirements.txt    # Dependency list
```


---
## 🔗 Data Source
Document: [Swiggy Annual Report FY 2023-24](https://drive.google.com/file/d/1yTooHqnyEzU1pI5fd6iK3VhRGMjpfAgc/view?usp=sharing)

---

## 📊 Evaluation Metrics
The system was evaluated using the **RAGAS framework**.
These results indicate that the generated responses remain highly grounded in the retrieved context while maintaining strong query relevance.
```text
| Metric           | Score |
|------------------|-------|
| Faithfulness     | 1.00  |
| Answer Relevancy | 0.89  |
```
---

## ⚙️ Setup and Installation
1. Clone the Repository:
```bash
git clone https://github.com/Aastha2675/RAG-Based_Questionary_answering_system.git
cd RAG-Based_Questionary_answering_system
```

2. Environment Setup:
Create a .env file in the root
```bash
HUGGINGFACE_API_KEY=your_huggingface_api_token
```

3. Run the Application:
```bash
streamlit run src/app.py
```

4. Run Evaluation:
```bash
python src/evaluate_rag.py
```

---

## 👤 Author
Aastha Mhatre 