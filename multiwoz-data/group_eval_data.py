import json
import ast

def group_data():
    input_file = "multiwoz_evaluation.json"
    output_file = "multiwoz_evaluation_grouped.json"
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        flat_data = json.load(f)
        
    grouped = {}
    
    for item in flat_data:
        d_id = item['dialogue_id']
        
        if d_id not in grouped:
            # Initialize the dialogue group
            # Parse services string back to list if possible, otherwise keep as is
            services_val = item['services']
            try:
                services_val = ast.literal_eval(services_val)
            except:
                pass

            grouped[d_id] = {
                "dialogue_id": d_id,
                "split": item['split'],
                "services": services_val,
                "turns": []
            }
        
        # Add the turn to the list
        turn_data = {
            "turn_id": item['turn_id'],
            "system_context": item['system_context'],
            "user_response_ground_truth": item['user_response_ground_truth']
        }
        grouped[d_id]["turns"].append(turn_data)
        
    # Convert dict to list
    grouped_list = list(grouped.values())
    
    print(f"Grouped {len(flat_data)} turns into {len(grouped_list)} dialogues.")
    
    with open(output_file, 'w') as f:
        json.dump(grouped_list, f, indent=2)
        
    print(f"Saved to {output_file}")
    
    # Preview
    print("\nFirst dialogue preview:")
    print(json.dumps(grouped_list[0], indent=2))

if __name__ == "__main__":
    group_data()
