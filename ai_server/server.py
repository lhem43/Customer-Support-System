# import
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import faiss
import numpy as np
import subprocess
import torch
import requests
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Configuration

SBERT_MODEL_PATH = "fine_tuned_kbqa_sbert"
HUGGING_FACE_API_KEY = ...
GPT2_MODEL_PATH = "fine_tuned_gpt2_v2"
INDEX_PATH = "faiss_index.bin"
ID_MAP_PATH = "faiss_id_map.npy"
WEB_SERVER_URL = "http://localhost:3333"
THRESHOLD = 0.35
global_vector_embedding = None
index = None
id_map = None

# init Flask
app = Flask(__name__)
CORS(app)

# Load SBert
sbert_model = SentenceTransformer(SBERT_MODEL_PATH)
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
print("Model loaded successfully!")

# Load FAISS Index & ID map
def load_index_map():
    global index, id_map
    try:
        index = faiss.read_index(INDEX_PATH)
        id_map = np.load(ID_MAP_PATH).tolist()
        print("FAISS and Map loaded successfully!")
    except:
        index = None
        id_map = []
        print("No FAISS found. Need initialization...")

# Synchronization
def get_max_faiss_id():
    """
    Return largest ID of FAISS in id_map
    """
    global id_map
    if not id_map:
        return 0
    return max(id_map)

def sync_faiss():
    """
    Synchronize FAISS with Postgres
    """
    global index
    global id_map
    try:
        response = requests.get(f"{WEB_SERVER_URL}/get_missing_questions", params={"max_faiss_id": get_max_faiss_id()}, timeout=10)
        response.raise_for_status()
        data = response.json()
        missing_questions = data.get("questions", [])
        missing_ids = data.get("ids", [])
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return
    try:
        response = requests.get(f"{WEB_SERVER_URL}/get_max_id", timeout=10)
        response.raise_for_status()
        max_postgres_id = response.json().get("max_id", 0)
    except requests.exceptions.RequestException as e:
        print(f"Error fetch max ID: {e}")
        return
    
    if max_postgres_id < get_max_faiss_id():
        print("Removing FAISS redundency.")
        remove_ids = [i for i in id_map if i > max_postgres_id]
        remove_indices = [id_map.index(i) for i in remove_ids]
        index.remove_ids(np.array(remove_indices, dtype=np.int64))

        id_map = [i for i in id_map if i <= max_postgres_id]

        faiss.write_index(index, INDEX_PATH)
        np.save(ID_MAP_PATH, np.array(id_map))
        load_index_map()
    if not missing_ids:
        print("FAISS is up to date.")
        return
    print (f"Updating FAISS with {len(missing_ids)} new questions.")
    new_embeddings = sbert_model.encode(missing_questions, convert_to_numpy=True)
    embedding_dim = new_embeddings.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(embedding_dim)
    
    index.add(new_embeddings)
    id_map.extend(missing_ids)
    try:
        faiss.write_index(index, INDEX_PATH)
        np.save(ID_MAP_PATH, np.array(id_map))
        print("FAISS syncing complete!")
    except Exception as e:
        print(f"Error saving FAISS: {e}")

@app.route("/query", methods=["POST"])
def query():
    """
    Handle query question from web server
    """
    global global_vector_embedding
    data =  request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    question_embedding = sbert_model.encode([question], convert_to_numpy=True)
    global_vector_embedding = question_embedding
    D, I = index.search(question_embedding, k=3)
    try:
        valid_ids = [id_map[I[0][i]] for i in range(len(I[0])) if D[0][i] <= THRESHOLD and I[0][i] != -1]
    except Exception as e:
        print(f"Error: {e}")
        valid_ids = []
    print(D[0][0])
    if not valid_ids:
        return jsonify({"answer": "I don't have a specific answer for this question. Please wait while I forward it to a support staff member."})
    else:
        try:
            responses = []
            for best_id in valid_ids:
                response = requests.get(f"{WEB_SERVER_URL}/get_answer", params={"id": best_id}, timeout=5)
                response.raise_for_status()
                answer = response.json().get("answer", "").strip()
                if answer:
                    responses.append(answer)
            
            if not responses:
                return jsonify({"answer": "I don't have a specific answer for this question. Please wait while I forward it to a support staff member."})
            answer = ". ".join(responses)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching answer: {e}")
            answer = "System error, please try again!"
    print(answer)
    last_answer = post_processing_answer(f"question: {question}, summarize this answer for that question: {answer}")
    print(last_answer)
    return jsonify({"answer": last_answer})

def summarize_context(answer):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"} 
    if "I don't have" in answer:
        return answer
    response = requests.post(API_URL, json={"inputs": answer}, headers=headers)
    
    if response.status_code == 200:
        summary = response.json()[0]["summary_text"]
        summary = summary.split(": ")[-1].strip()
        return summary
    else:
        print("Error:", response.text)
        return answer

def post_processing_answer(answer):
    """
    GPT2 post-processing answer
    """
    answer = summarize_context(answer)
    prompt = f"sentence: {answer}|paraphrase:"
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        output = gpt2_model.generate(
            input_ids,
            max_length=1024,
            num_return_sequences=2,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    if "paraphrase:" in generated_text:
        clean_answer = generated_text.split("paraphrase:")[-1].strip()
    else:
        clean_answer = generated_text
    return clean_answer

@app.route("/update_faiss", methods=["POST"])
def update_faiss():
    global global_vector_embedding
    data = request.json
    id = data.get("id")
    question = data.get("question")
    if not question or id is None:
        return jsonify({"error": "Missing parameters"}), 400
    
    if global_vector_embedding is None:
        question_embedding = sbert_model.encode([question], convert_to_numpy=True)
    else:
        question_embedding = global_vector_embedding
        global_vector_embedding = None
    index.add(question_embedding)
    id_map.append(id)
    faiss.write_index(index, "faiss_index.bin")
    np.save("faiss_id_map.npy", np.array(id_map))
    load_index_map()
    return jsonify({"message": "FAISS index updated successfully", "id": id})

@app.route("/")
def home():
    return redirect("http://localhost:8501")

load_index_map()
sync_faiss() # start syncing
subprocess.Popen(["streamlit", "run", "metrics.py"])
app.run(host="0.0.0.0", port=7890, debug=False)