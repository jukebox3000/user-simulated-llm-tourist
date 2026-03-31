import csv
import os
from multi import load_local_multiwoz
import pandas as pd

def create_eval_data():
    # 1. Load Data
    data_path = "./multiwoz_data/data/MultiWOZ_2.2"
    dataset = load_local_multiwoz(data_path)
    
    flattened_data = []
    
    # 2. Iterate through specific splits (using dev and test for evaluation as requested)
    # The user said they just need "data for evaluation", combining dev and test gives a good amount.
    target_splits = ['validation', 'test'] 
    
    for split in target_splits:
        if split not in dataset:
            continue
            
        print(f"Processing {split}...")
        for dialogue in dataset[split]:
            dialogue_id = dialogue['dialogue_id']
            turns = dialogue['turns']
            services = dialogue.get('services', [])
            
            # We want pairs of (System Context) -> (User Response)
            # The first turn is always User. Context is empty or "start".
            
            last_system_utterance = "<START_OF_DIALOGUE>"
            
            for turn in turns:
                speaker = turn['speaker']
                utterance = turn['utterance']
                
                if speaker == "SYSTEM":
                    last_system_utterance = utterance
                elif speaker == "USER":
                    # This is a sample we can evaluate on.
                    # Given the last system message, what did the user say?
                    
                    row = {
                        'dialogue_id': dialogue_id,
                        'turn_id': turn['turn_id'],
                        'split': split,
                        'services': str(services),
                        'system_context': last_system_utterance,
                        'user_response_ground_truth': utterance
                    }
                    flattened_data.append(row)

    # 3. Save to CSV
    output_file_csv = "multiwoz_evaluation.csv"
    df = pd.DataFrame(flattened_data)
    df.to_csv(output_file_csv, index=False)
    print(f"Successfully saved {len(df)} evaluation pairs to {output_file_csv}")

    # 4. Save to JSON
    output_file_json = "multiwoz_evaluation.json"
    import json
    with open(output_file_json, 'w') as f:
        json.dump(flattened_data, f, indent=2)
    print(f"Successfully saved {len(flattened_data)} evaluation pairs to {output_file_json}")
    
    # 5. Preview
    print("\nFirst 3 rows:")
    print(df.head(3))

if __name__ == "__main__":
    create_eval_data()
