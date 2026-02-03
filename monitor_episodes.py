import json
import time
import os
import glob
from datetime import datetime
from collections import deque
import numpy as np
import traceback

# Try importing SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: sentence-transformers or scikit-learn not installed.")
    print("Please run: pip install sentence-transformers scikit-learn")
    exit(1)

# Configuration
BLOCKS_DIR = "./Factory-4/blocks"
PLAYBACK_SPEED = 0.5
MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.8
LIFE_THREATENING_THRESHOLD = 0.6
FIFO_SIZE = 10

# State Machine Constants
STATE_IDLE = "IDLE"
STATE_IN_EPISODE = "IN_EPISODE"
STATE_PENDING_CLOSURE = "PENDING_CLOSURE"

# Thresholds
START_THRESHOLD = 60.0
END_THRESHOLD = 40.0

# --- LLM Integration (Placeholder) ---
def call_gemma_llm(context_data):
    """
    Summarizes the provided context data using Gemma 3 logic.
    Accepts the full list of JSON blocks for the episode.
    """
    frame_count = len(context_data)
    print(f"\n[System] Generating Summary via Gemma 3 for {frame_count} frames...")
    return f"Gemma 3 Summary: Episode with {frame_count} frames processed. Triggered by sensor thresholds or semantic risk."

# --- Helper Functions ---

def extract_numerical_variables(data, parent_key=''):
    """
    Recursively extracts all numerical variables (int, float, or string-numbers)
    from a dictionary and returns a flattened dict of {path: value}.
    """
    items = {}
    
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            items.update(extract_numerical_variables(v, new_key))
            
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            items.update(extract_numerical_variables(v, new_key))
            
    else:
        if isinstance(data, (int, float)) and not isinstance(data, bool):
             if data <= 100.0:
                 items[parent_key] = data
        elif isinstance(data, str):
            try:
                val = float(data)
                if val <= 100.0:
                    items[parent_key] = val
            except ValueError:
                pass
                
    return items

def get_embedding(model, text):
    if not text:
        return np.zeros(384) 
    return model.encode(text)

def compute_similarity(emb1, emb2):
    """Computes cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return 0.0
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

def check_new_episode_trigger(max_val, current_embedding, life_threatening_embedding):
    """
    Checks if a new episode should start based on:
    1. Variable Threshold (max_val > START_THRESHOLD)
    2. Semantic Similarity to 'life threatening' (sim > LIFE_THREATENING_THRESHOLD)
    
    Returns: (is_triggered, reason_string)
    """
    # Check 1: Variable Threshold
    if max_val > START_THRESHOLD:
        return True, f"High Variable ({max_val:.2f})"
        
    # Check 2: Life Threatening Semantic
    lt_sim_score = compute_similarity(current_embedding, life_threatening_embedding)
    if lt_sim_score > LIFE_THREATENING_THRESHOLD:
        return True, f"Semantic Threat ({lt_sim_score:.3f})"
        
    return False, ""

def finalize_episode(episode_buffer, reason_suffix="Ended"):
    """
    Generates summary for the episode buffer and prints it.
    Returns empty buffer to reset state.
    """
    if not episode_buffer:
        return []

    print("-" * 40)
    summary = call_gemma_llm(episode_buffer)
    print(f"EPISODE SUMMARY ({reason_suffix}):")
    print(summary)
    print("-" * 40)
    return []

# --- Main Execution ---

def main():
    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    # Pre-compute "life threatening" embedding
    life_threatening_text = "life threatening"
    life_threatening_embedding = get_embedding(model, life_threatening_text)
    print(f"Computed embedding reference for '{life_threatening_text}'")

    current_state = STATE_IDLE
    episode_buffer = []          
    pending_buffer = []          
    
    recent_embeddings = deque(maxlen=FIFO_SIZE) 
    last_active_embedding = None 

    search_path = os.path.join(BLOCKS_DIR, "block_*.json")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No files found in {search_path}")
        return

    print(f"Found {len(files)} blocks to process.")
    print(f"Monitoring... (Sim > {SIMILARITY_THRESHOLD}, Start > {START_THRESHOLD}, LifeThreat > {LIFE_THREATENING_THRESHOLD})")

    for file_path in files:
        try:
            print(f"Processing {os.path.basename(file_path)}...", end='\r')
            
            with open(file_path, 'r') as f:
                data = json.load(f)

            data['source_file'] = os.path.basename(file_path)
            
            # 1. Get Summary & Embedding
            summary_content = data.get('summary', {})
            summary_text = json.dumps(summary_content)
            current_embedding = get_embedding(model, summary_text)
            recent_embeddings.append(current_embedding)

            # 2. Extract Variables
            flat_vars = extract_numerical_variables(summary_content, parent_key='summary')
            max_val = 0.0
            if flat_vars:
                trigger_var = max(flat_vars, key=flat_vars.get)
                max_val = flat_vars[trigger_var]
            
            # 3. State Machine Logic
            
            if current_state == STATE_IN_EPISODE:
                # OPTIMIZATION: Skip similarity check
                episode_buffer.append(data)
                last_active_embedding = current_embedding 

                # Check Life Threatening for "Hold" condition
                lt_sim_score = compute_similarity(current_embedding, life_threatening_embedding)

                if max_val < END_THRESHOLD and lt_sim_score < LIFE_THREATENING_THRESHOLD:
                    print(f"\n[Info] Value dropped ({max_val:.2f}) and Risk low ({lt_sim_score:.3f}). Checking continuity...")
                    current_state = STATE_PENDING_CLOSURE
                    pending_buffer = [] 
                else:
                    trigger_msg = f"Var: {max_val:.1f}" if max_val >= END_THRESHOLD else f"Risk: {lt_sim_score:.3f}"
                    print(f"In Episode... {trigger_msg} (Buffer: {len(episode_buffer)})", end='\r')

            elif current_state == STATE_PENDING_CLOSURE:
                pending_buffer.append(data)
                
                # Check Similarity to Previous Episode
                sim_score = compute_similarity(current_embedding, last_active_embedding)
                
                # Check New Trigger (Var or LifeThreat)
                is_triggered, trigger_reason = check_new_episode_trigger(max_val, current_embedding, life_threatening_embedding)

                if sim_score > SIMILARITY_THRESHOLD:
                    # RESUME
                    print(f"\n[Info] High Similarity ({sim_score:.3f}) detected! Resuming episode.")
                    episode_buffer.extend(pending_buffer)
                    last_active_embedding = current_embedding 
                    pending_buffer = []
                    current_state = STATE_IN_EPISODE

                elif is_triggered:
                    # NEW EPISODE
                    print(f"\n[Info] {trigger_reason} detected during gap. Starting NEW episode.")
                    
                    # Finalize Old
                    episode_buffer = finalize_episode(episode_buffer, reason_suffix="Ended")
                    
                    # Start New
                    episode_buffer = [data]
                    last_active_embedding = current_embedding
                    pending_buffer = []
                    current_state = STATE_IN_EPISODE

                elif len(pending_buffer) >= FIFO_SIZE:
                    # TIMEOUT
                    print(f"\n[Info] Gap timeout ({len(pending_buffer)} frames). Finalizing episode.")
                    episode_buffer = finalize_episode(episode_buffer, reason_suffix="Ended")
                    
                    pending_buffer = []
                    last_active_embedding = None
                    current_state = STATE_IDLE

            elif current_state == STATE_IDLE:
                is_triggered, trigger_reason = check_new_episode_trigger(max_val, current_embedding, life_threatening_embedding)
                
                if is_triggered:
                    print(f"\n[!!!] EPISODE STARTED in {data['source_file']}") 
                    print(f"      Trigger: {trigger_reason}")
                    
                    current_state = STATE_IN_EPISODE
                    episode_buffer = [data]
                    last_active_embedding = current_embedding
                
            # time.sleep(PLAYBACK_SPEED)

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            traceback.print_exc()

    # End of files check
    if current_state in [STATE_IN_EPISODE, STATE_PENDING_CLOSURE]:
         if episode_buffer:
             print(f"\n[!] Script finished. Finalizing remaining episode.")
             finalize_episode(episode_buffer, reason_suffix="Final")

if __name__ == "__main__":
    main()
