
import json
import time
import random
import os

JSON_FILE_PATH = "sample_output.json"

def write_json(risk_value):
    data = {
        "ts": time.time(),
        "summary": {
            "Risk": "Test Risk",
            "CollisionRisk": risk_value, # This is our trigger variable
            "OtherMetric": 10
        }
    }
    
    with open(JSON_FILE_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated {JSON_FILE_PATH} with CollisionRisk = {risk_value}")

def main():
    print("Starting Data Generator Simulation...")
    
    # 1. Idle State
    print("Phase 1: IDLE (Values < 60)")
    vals = [10, 20, 30, 40, 50]
    for v in vals:
        write_json(v)
        time.sleep(1.2)
        
    # 2. Trigger Episode
    print("\nPhase 2: TRIGGER EPISODE (Value > 60)")
    write_json(75)
    time.sleep(1.2)
    write_json(80)
    time.sleep(1.2)
    
    # 3. Sustain Episode
    print("\nPhase 3: SUSTAIN EPISODE (Value > 40)")
    vals = [70, 65, 55, 45]
    for v in vals:
        write_json(v)
        time.sleep(1.2)
        
    # 4. End Episode
    print("\nPhase 4: END EPISODE (Value < 40)")
    write_json(35)
    time.sleep(1.2)
    write_json(20)
    
    print("\nSimulation Complete.")

if __name__ == "__main__":
    main()
