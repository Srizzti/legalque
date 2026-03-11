import os
import random
import google.generativeai as genai
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

YOUR_API_KEY = "changeme"
genai.configure(api_key=YOUR_API_KEY)
generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)

db_path =change-me
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


def generate_queries_gemini(original_query):
    content_prompts = [f"Generate a single refined query related to: {original_query} with a proper description in 4-5 lines."]
    response = model.generate_content(content_prompts)
    return response.text.strip()

def vector_search(query, cursor):
    query_vector = np.random.rand(1, 384).astype(np.float32)
    cursor.execute("SELECT filename, vector FROM documents")
    rows = cursor.fetchall()
    scores = {}
    for filename, vector_blob in rows:
        vector = np.frombuffer(vector_blob, dtype=np.float32)
        vector = vector.reshape(1, -1)
        
        similarity_score = cosine_similarity(query_vector, vector)[0][0]
        scores[filename] = similarity_score

    return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > 0}

def reciprocal_rank_fusion(search_results_dict, k=60):
    try:
        fused_scores = {}
        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                fused_scores[doc] += 1 / (rank + k)
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}

        print("\nReranked Results (Document Importance Scores):")
        for doc, score in reranked_results.items():
            print(f"Document: {doc}, Importance Score: {score:.4f}")

        return reranked_results
    except Exception as e:
        print(f"Error performing Reciprocal Rank Fusion: {e}")
        return {}


def generate_content_google(reranked_results, queries):
    content_prompts = [f"Summarize the key information from the following queries and documents in exactly four concise lines: {queries} and {list(reranked_results.keys())}"]
    response = model.generate_content(content_prompts)
    
    response_lines = response.text.strip().split("\n")
    summarized_response = " ".join(response_lines).split(". ") 
    return ". ".join(summarized_response[:4]) + ('.' if len(summarized_response) >= 4 else '')

if __name__ == "__main__":
    original_query = str(input("Enter the query:-"))
    generated_queries = generate_queries_gemini(original_query)

    all_results = {}
    for query in generated_queries:
        search_results = vector_search(query, cursor)
        all_results[query] = search_results

    reranked_results = reciprocal_rank_fusion(all_results)
    
    final_output = generate_content_google(reranked_results, generated_queries)
    print("\n \n")
    print(final_output)
    conn.close()
