import json
import glob
import os

def load_local_multiwoz(data_root):
    dataset = {}
    
    # Map raw folder names to standard split names
    splits = {
        'train': 'train',
        'dev': 'validation', 
        'test': 'test'
    }
    
    for folder, split_name in splits.items():
        path = os.path.join(data_root, folder)
        if not os.path.exists(path):
            print(f"Warning: Path not found {path}")
            continue
            
        all_dialogues = []
        # Find all json files like dialogues_001.json
        files = glob.glob(os.path.join(path, "*.json"))
        
        for f in sorted(files):
            with open(f, 'r') as fd:
                data = json.load(fd)
                # data is a list of dialogues
                all_dialogues.extend(data)
                
        dataset[split_name] = all_dialogues
        print(f"Loaded {len(all_dialogues)} dialogues for {split_name}")
        
    return dataset

# Path to the cloned repo data
data_path = "./multiwoz_data/data/MultiWOZ_2.2"
dataset = load_local_multiwoz(data_path)

# Mocking the huggingface dataset structure slightly for print compatibility if needed, 
# but for now it returns a dict of lists which is close enough for inspection.
print(f"Dataset keys: {dataset.keys()}")
if 'train' in dataset:
    print(f"First training example ID: {dataset['train'][0]['dialogue_id']}")