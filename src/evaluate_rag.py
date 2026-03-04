import os
import warnings
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv

# import your RAG pipeline
from rag_pipeline import generate_answer, load_vectorstore


warnings.filterwarnings("ignore")
load_dotenv()


# -----------------------------
# Setup Judge LLM + Embeddings
# -----------------------------
def setup_judge():

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        temperature=0.01,
        max_new_tokens=2048
    )

    chat_model = ChatHuggingFace(llm=llm)
    judge_llm = LangchainLLMWrapper(chat_model)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    judge_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return judge_llm, judge_embeddings


# -----------------------------
# Run Evaluation
# -----------------------------
def run_evaluation():

    print("Loading vector database...")
    vectordb = load_vectorstore()

    judge_llm, judge_embeddings = setup_judge()

    # Evaluation Questions
    test_questions = [
        "What was the total income of Swiggy for the year ended March 31, 2024?",
        "How does the report describe Swiggy's mission and core business?",
        "What are the key financial highlights mentioned in the report summary?"
    ]

    results = []

    print("\nGenerating RAG answers...\n")

    for q in test_questions:

        try:

            answer, context, citations = generate_answer(q, vectordb)

            results.append({
                "question": q,
                "answer": str(answer),
                "contexts": [str(context)]
            })

            print(f"Generated answer for: {q[:60]}")

        except Exception as e:
            print(f"Generation failed: {e}")

    # Convert to dataset
    dataset = Dataset.from_list(results)

    print("\nRunning RAGAS Evaluation...\n")

    metrics = [
        faithfulness,
        answer_relevancy
    ]

    try:

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False
        )

        print("\n" + "="*60)
        print("SWIGGY RAG EVALUATION REPORT")
        print("="*60)

        # Convert to dataframe
        df_scores = result.to_pandas()

        # Calculate average metrics
        faithfulness_score = df_scores["faithfulness"].dropna().mean()
        relevancy_score = df_scores["answer_relevancy"].dropna().mean()

        print("\nAverage Scores:\n")
        print({
            "faithfulness": round(faithfulness_score,4),
            "answer_relevancy": round(relevancy_score,4)
        })

        # Create clean dataframe
        final_scores = pd.DataFrame([{
            "faithfulness": round(faithfulness_score,4),
            "answer_relevancy": round(relevancy_score,4)
        }])

        # Save results
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_path = os.path.join(ROOT_DIR, "model_evaluation")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"\nCreated folder: {folder_path}")

        csv_path = os.path.join(folder_path, "evaluation_results.csv")

        final_scores.to_csv(csv_path, index=False)

        print(f"\nEvaluation results saved to:\n{csv_path}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"\nEvaluation failed: {e}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_evaluation()