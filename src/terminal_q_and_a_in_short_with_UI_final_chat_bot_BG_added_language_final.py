import gradio as gr
import sqlite3
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import csv
import os

genai.configure(api_key="")
generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)

db_path = change-me
csv_file_path = change-me

if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["User Query", "Bot Response"])

def log_chat_to_csv(user_query, bot_response):
    """Append each query-response pair to a CSV file."""
    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([user_query, bot_response])

def generate_queries_gemini(original_query):
    content_prompts = [f"Generate a single refined query related to: {original_query} with a proper description in 4-5 lines."]
    response = model.generate_content(content_prompts)
    return response.text.strip()

def vector_search(query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query_vector = np.random.rand(1, 384).astype(np.float32)
    cursor.execute("SELECT filename, vector FROM documents")
    rows = cursor.fetchall()
    scores = {}
    for filename, vector_blob in rows:
        vector = np.frombuffer(vector_blob, dtype=np.float32).reshape(1, -1)
        similarity_score = cosine_similarity(query_vector, vector)[0][0]
        scores[filename] = similarity_score
    conn.close()
    return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > 0}

def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (rank + k)
    return {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}

def generate_content_google(reranked_results, queries):
    content_prompts = [f"Summarize the key information from the following queries and documents in exactly four concise lines: {queries} and {list(reranked_results.keys())}"]
    response = model.generate_content(content_prompts)
    response_lines = response.text.strip().split("\n")
    summarized_response = " ".join(response_lines).split(". ")
    return ". ".join(summarized_response[:4]) + ('.' if len(summarized_response) >= 4 else '')

def translate_text(text, target_language):
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  

def chatbot_interface(message, chat_history, language):
    if chat_history is None:
        chat_history = []  
    
    chat_history.append({"role": "user", "content": message})
    generated_queries = generate_queries_gemini(message)
    all_results = {query: vector_search(query) for query in generated_queries.split('\n')}
    reranked_results = reciprocal_rank_fusion(all_results)
    response = generate_content_google(reranked_results, generated_queries)
    
    if language == "Hindi": 
        response = translate_text(response, "hi")
    elif language == "Tamil":  
        response = translate_text(response, "ta")
    log_chat_to_csv(message, response)

    chat_history.append({"role": "assistant", "content": response})
    
    return chat_history, chat_history

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="Enter your query"),
        gr.State(),
        gr.Radio(["English", "Hindi","Tamil"], label="Select Response Language", value="English")
    ],
    outputs=[
        gr.Chatbot(type="messages", elem_id="chatbot-container")
,
        gr.State()
    ],
    title="RAG-Based Gemini Chatbot",
    description="Chat with a RAG-powered Gemini search bot that refines queries, retrieves documents, and summarizes insights. Select whether you want the response in English,Hindi or Tamil",
    css="""/* Add your CSS here */"""
)

iface.launch(show_error=True)

