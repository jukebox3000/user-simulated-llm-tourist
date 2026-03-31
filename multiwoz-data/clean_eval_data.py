import json

def clean_data():
    input_file = "multiwoz_evaluation_grouped.json"
    output_file = "cleaned.json"
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        grouped_data = json.load(f)
        
    cleaned_dialogues = []
    
    for dialogue in grouped_data:
        # Create a new structure for the dialogue
        new_dialogue = {
            "dialogue_id": dialogue['dialogue_id'],
            "split": dialogue['split'],
            "services": dialogue['services'],
            "turns": []
        }
        
        # Unroll the paired turns into sequential separate objects
        paired_turns = dialogue['turns']
        
        for turn in paired_turns:
            sys = turn['system_context']
            usr = turn['user_response_ground_truth']
            turn_id = turn['turn_id'] # Use the ID from the source pair for reference if needed
            
            # Add System turn if it's not the start token
            if sys != "<START_OF_DIALOGUE>":
                new_dialogue['turns'].append({
                    "speaker": "SYSTEM",
                    "utterance": sys
                })
                
            # Add User turn (always present in the pair)
            new_dialogue['turns'].append({
                "speaker": "USER",
                "utterance": usr
            })
            
        cleaned_dialogues.append(new_dialogue)
    
    print(f"Processed {len(cleaned_dialogues)} dialogues.")
    
    with open(output_file, 'w') as f:
        json.dump(cleaned_dialogues, f, indent=2)
        
    print(f"Saved to {output_file}")
    
    # Preview
    print("\nFirst dialogue preview:")
    print(json.dumps(cleaned_dialogues[0], indent=2))

if __name__ == "__main__":
    clean_data()
