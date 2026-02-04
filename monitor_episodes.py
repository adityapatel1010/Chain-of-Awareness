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
BLOCKS_DIR = "./shooting-1/blocks"
PLAYBACK_SPEED = 0.5
MODEL_NAME = 'all-MiniLM-L6-v2'
FIFO_SIZE = 10

# Semantic Configuration (List of texts and corresponding thresholds)
SEMANTIC_TEXTS = ["life threatening", "gun shooting","physically assaulting"]
SEMANTIC_THRESHOLDS = [0.6, 0.6,0.6]

# State Machine Constants
STATE_IDLE = "IDLE"
STATE_IN_EPISODE = "IN_EPISODE"
STATE_PENDING_CLOSURE = "PENDING_CLOSURE"

# Thresholds
SIMILARITY_THRESHOLD = 0.8 # For Gap Resilience (similarity to prev episode)
START_THRESHOLD = 60.0 # Variable Start
END_THRESHOLD = 40.0   # Variable End

# --- LLM Integration (Placeholder) ---
def call_gemma_llm(context_data):
    """
    Summarizes the provided context data.
    """
    frame_count = len(context_data)
    print(f"\n[System] Generating Summary via Gemma 3 for {frame_count} frames...")
    return f"Gemma 3 Summary: Episode with {frame_count} frames processed. Triggered by sensor or semantic risk."

# --- Helper Functions ---

def extract_numerical_variables(data, parent_key=''):
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
    if emb1 is None or emb2 is None:
        return 0.0
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

def check_new_episode_trigger(max_val, current_embedding, trigger_embeddings, thresholds):
    """
    Checks if a new episode should start.
    Returns: (is_triggered, reason_string)
    """
    # 1. Variable Trigger
    if max_val > START_THRESHOLD:
        return True, f"High Variable ({max_val:.2f})"
        
    # 2. Semantic Triggers
    if not trigger_embeddings or not thresholds:
        return False, ""
        
    for i, t_emb in enumerate(trigger_embeddings):
        sim = compute_similarity(current_embedding, t_emb)
        title = SEMANTIC_TEXTS[i] if i < len(SEMANTIC_TEXTS) else f"Trigger {i}"
        
        print(f"      [SimCheck] '{title}': {sim:.4f}")
        
        if sim > thresholds[i]:
            return True, f"Semantic '{title}' ({sim:.3f})"
            
    return False, ""

def check_semantic_hold(current_embedding, trigger_embeddings, thresholds):
    """
    Checks if ANY semantic trigger is high enough to HOLD the episode active.
    Returns: (should_hold, reason_string)
    """
    if not trigger_embeddings or not thresholds:
        return False, ""

    # Check matches
    for i, t_emb in enumerate(trigger_embeddings):
        sim = compute_similarity(current_embedding, t_emb)
        title = SEMANTIC_TEXTS[i] if i < len(SEMANTIC_TEXTS) else f"Trigger {i}"
        
        print(f"      [HoldCheck] '{title}': {sim:.4f}")
        
        if sim >= thresholds[i]:
            return True, f"Risk: '{title}' ({sim:.3f})"
            
    return False, ""

def finalize_episode(episode_buffer, reason_suffix="Ended"):
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

    # Pre-compute "trigger" embeddings
    trigger_embeddings = []
    print("Computing trigger embeddings:")
    for i, text in enumerate(SEMANTIC_TEXTS):
        emb = get_embedding(model, text)
        trigger_embeddings.append(emb)
        print(f" - [{i}] '{text}' (Threshold: {SEMANTIC_THRESHOLDS[i]})")

    current_state = STATE_IDLE
    episode_buffer = []          
    pending_buffer = []          
    
    last_active_embedding = None 

    # Look for block_*.json 
    search_path = os.path.join(BLOCKS_DIR, "block_*.json")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No files found in {search_path}")
        return

    print(f"Found {len(files)} blocks to process.")
    print("Monitoring...")

    for file_path in files:
        # FILTER: Skip if ends with _state.json
        if file_path.endswith("_state.json"):
            continue

        try:
            print(f"Processing {os.path.basename(file_path)}...", end='\r')
            
            with open(file_path, 'r') as f:
                data = json.load(f)

            data['source_file'] = os.path.basename(file_path)
            
            # 1. Get Summary & Embedding
            summary_content = data.get('summary', {})
            summary_text = json.dumps(summary_content)
            current_embedding = get_embedding(model, summary_text)
            
            # 2. Extract Variables
            flat_vars = extract_numerical_variables(summary_content, parent_key='summary')
            max_val = 0.0
            if flat_vars:
                trigger_var = max(flat_vars, key=flat_vars.get)
                max_val = flat_vars[trigger_var]
            
            # 3. State Machine Logic
            
            if current_state == STATE_IN_EPISODE:
                episode_buffer.append(data)
                last_active_embedding = current_embedding 

                # Check "Hold" Conditions (Variable OR Semantic)
                var_hold = (max_val >= END_THRESHOLD)
                sem_hold, sem_msg = check_semantic_hold(current_embedding, trigger_embeddings, SEMANTIC_THRESHOLDS)
                
                if not var_hold and not sem_hold:
                    # DROP Condition Met
                    print(f"\n[Info] Value dropped ({max_val:.2f}) and No Semantic Risk. checking continuity...")
                    current_state = STATE_PENDING_CLOSURE
                    pending_buffer = [] 
                else:
                    # Stay Active
                    msg = f"Var: {max_val:.1f}" if var_hold else sem_msg
                    print(f"In Episode... {msg} (Buffer: {len(episode_buffer)})", end='\r')

            elif current_state == STATE_PENDING_CLOSURE:
                pending_buffer.append(data)
                
                # Check Similarity to Previous Episode
                sim_score = compute_similarity(current_embedding, last_active_embedding)
                
                # Check New Trigger
                is_triggered, trigger_reason = check_new_episode_trigger(max_val, current_embedding, trigger_embeddings, SEMANTIC_THRESHOLDS)

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
                    episode_buffer = finalize_episode(episode_buffer, reason_suffix="Ended")
                    
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
                is_triggered, trigger_reason = check_new_episode_trigger(max_val, current_embedding, trigger_embeddings, SEMANTIC_THRESHOLDS)
                
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

    # End
    if current_state in [STATE_IN_EPISODE, STATE_PENDING_CLOSURE]:
         if episode_buffer:
             print(f"\n[!] Script finished. Finalizing remaining episode.")
             finalize_episode(episode_buffer, reason_suffix="Final")

if __name__ == "__main__":
    main()
