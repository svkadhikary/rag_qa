# evaluate.py
import json
import re
import asyncio
import requests
from datetime import datetime
import pandas as pd
# from datasets import Dataset
# from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
# from ragas.llms import llm_factory
# from ragas.metrics._faithfulness import Faithfulness
# from ragas.metrics._answer_relevance import AnswerRelevancy
# from ragas.metrics._context_precision import ContextPrecision
# from ragas.metrics._context_recall import ContextRecall

# from ragas.metrics.collections import (
#     Faithfulness,
#     AnswerRelevancy,
#     ContextRecall,
#     ContextPrecision
#     )

# Import your existing RAG pipeline components directly
from utils import load_and_split_pdf, text_splitter
from TransformerUtils.transformer_utils import transform_sentences
from VectorStores.vectorstore import VectorStore, search
from llms.reranker import cohere_rerank
from llms.model_calls import GroqChatModel

TEST_FILE = "test_dataset.json"

# - Create dataset for testing
test_doc = ""
with open("attention.txt", 'r', encoding='utf-8') as f:
    test_doc = f.read()

# extracted_data = GroqChatModel(model_name="openai/gpt-oss-20b", temperature=0).create_test_qa(test_doc)
# with open(TEST_FILE, 'w', encoding='utf-8') as f:
#     json.dump(extracted_data, f, indent=4, ensure_ascii=False)
# print(f"Successfully saved {len(extracted_data['dataset'])} QA pairs to {TEST_FILE}!")

# ── 1. Load your test dataset ──────────────────────────────────────────────────
# Format: list of dicts with keys: question, ground_truth
with open("test_dataset.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# ── 2. Build the index once (point to a test PDF or pre-built index) ───────────
# PDF_PATH = "your_test_document.pdf"

# chunks = load_and_split_pdf(PDF_PATH, chunk_size=400, overlap=50)
# vectors = transform_sentences(chunks)
# index = VectorStore(vectors.shape[1], chunks[0][:10]).create_index(vectors)

## OR

# URLs
# urls = ["https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"]
# temp_text_list = [] 
    
# for url in urls:
#     try:
#         res = requests.get(url)
#         temp_text_list.append(res.text)
#     except Exception as e:
#         raise(str(e))

# # Combine text for processing
# full_text = "\n".join(temp_text_list)

# Process chunks
chunks = text_splitter(test_doc, 500, 50)
vectors = transform_sentences(chunks)
# Create vector store index
index = VectorStore(vectors.shape[1], chunks[0][:10]).create_index(vectors)


# ── 3. Run your RAG pipeline for each question ─────────────────────────────────
def run_rag_pipeline(question: str):
    enriched_query = GroqChatModel(temperature=0.5).query_rewrite(question)

    query_vector = transform_sentences([enriched_query])
    distances, indices = search(query_vector, index, top_k=10)
    relevant_indices = indices[0]

    relevant_chunks = [chunks[i] for i in relevant_indices if i != -1]
    relevant_chunks = cohere_rerank(enriched_query, relevant_chunks, top_k=5)

    answer = GroqChatModel().generate_responses(enriched_query, ''.join(relevant_chunks))
    return answer, relevant_chunks

# ── 4. Collect results into RAGAS format ──────────────────────────────────────

samples = []
for item in test_data['dataset']:
    answer, context_chunks = run_rag_pipeline(item["question"])
    # Extract the string text from LangChain's AIMessage safely
    if hasattr(answer, 'content'):
        clean_answer = answer.content
    else:
        clean_answer = str(answer)

    samples.append({
        "query": item["question"],
        "response": clean_answer,
        "contexts": [str(c) for c in context_chunks],
        "ground_truth": item["ground_truth"]
    })

# Define evaluation prompts
EVAL_PROMPT = """
You are an expert AI Quality Assurance Judge. Evaluate the following RAG system outputs.

[User Query]: {query}
[Retrieved Context]: {context}
[System Response]: {response}
[Ground Truth]: {ground_truth}

Provide two scores between 0.0 and 1.0:
1. Faithfulness: Is the System Response completely grounded in and supported ONLY by the Retrieved Context? (1.0 = fully grounded, 0.0 = contains hallucinations or outside info).
2. Context Recall: Does the Retrieved Context contain the necessary information to address the Ground Truth? (1.0 = contains all required info, 0.0 = missed the answer completely).

Your output must follow this exact JSON format:
{{
    "faithfulness": A score between 0 to 1,
    "context_recall": A score between 0 to 1,
    "reasoning": "Explain your score choice here.."
}}
"""

def evaluate_rag_custom(data_samples, judge_model):
    results = []
    
    for i, sample in enumerate(data_samples):
        print(f"Evaluating sample {i+1}/{len(data_samples)}...")
        
        context_str = "\n---\n".join(sample["contexts"])
        prompt = EVAL_PROMPT.format(
            query=sample["query"],
            context=context_str,
            response=sample["response"],
            ground_truth=sample["ground_truth"]
        )
        
        try:
            # Generate the grading text using your Groq wrapper
            raw_eval = judge_model.invoke(prompt)
            
            # Extract content text if the judge model also outputs an AIMessage object
            clean_eval = raw_eval.content if hasattr(raw_eval, 'content') else str(raw_eval)
            
            # Extract the raw JSON block out of the response text
            json_match = re.search(r"\{.*\}", clean_eval, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON structure found in output text.")
                
            results.append({
                "question": sample["query"],
                "response": sample["response"],
                "ground_truth": sample["ground_truth"],
                "faithfulness": scores.get("faithfulness", None),
                "context_recall": scores.get("context_recall", None),
                "reasoning": scores.get("reasoning", "")
            })
            
        except Exception as e:
            print(f"Error evaluating sample {i+1}: {str(e)}")
            results.append({
                "question": sample["query"],
                "response": sample["response"],
                "ground_truth": sample["ground_truth"],
                "faithfulness": None,
                "context_recall": None,
                "reasoning": f"Evaluation Broken: {str(e)}"
            })
            
    return pd.DataFrame(results)


# ── 5. RUN CUSTOM EVALUATION ──────────────────────────────────────────────────
# Initialize your existing Groq wrapper to act as the Judge
judge_llm = GroqChatModel(temperature=0).client

# Run the custom evaluator on your collected samples list
df_scores = evaluate_rag_custom(samples, judge_llm)

# Compute global averages while safely skipping any errors (NaNs)
print("\n--- Final Evaluation Metrics ---")
print(f"Average Faithfulness: {df_scores['faithfulness'].mean():.2f}")
print(f"Average Context Recall: {df_scores['context_recall'].mean():.2f}")

# Save results exactly like before
file_name = f"custom_scores_{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"
df_scores.to_csv(f"eval_results/{file_name}", index=False)
print(f"Saved evaluation results to eval_results/{file_name}")

# evaluation_dataset = EvaluationDataset(samples=samples)

# results = {
#     "question": [],
#     "answer": [],
#     "contexts": [],
#     "ground_truth": []
# }

# for item in test_data['dataset']:
#     question = item["question"]
#     ground_truth = item["ground_truth"]

#     answer, context_chunks = run_rag_pipeline(question)

#     results["question"].append(question)
#     results["answer"].append(answer)
#     results["contexts"].append(context_chunks)   # list of strings per question
#     results["ground_truth"].append(ground_truth)

# groq_llm = GroqChatModel().client
# # evaluator_llm = llm_factory(model=groq_llm.model_name, client=groq_llm.client)
# # ── 5. Run RAGAS evaluation ───────────────────────────────────────────────────
# # dataset = Dataset.from_dict(results)

# evaluation_metrics = [
#         Faithfulness(llm=evaluator_llm),           # is the answer grounded in the context?
#         ContextPrecision(llm=evaluator_llm),      # are retrieved chunks actually useful?
#         ContextRecall(llm=evaluator_llm)         # did retrieval capture what was needed?
#     ]

# scores = evaluate(
#     dataset = evaluation_dataset,
#     metrics = evaluation_metrics,
# )

# print(f"Scores:{scores}")
# file_name = f"ragas_scores_{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"
# scores.to_pandas().to_csv(f"eval_results/{file_name}", index=False)