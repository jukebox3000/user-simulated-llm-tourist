from multi import load_local_multiwoz
import pandas as pd

# -------- 1. Load MultiWOZ 2.2 from Local Data --------
# Using the function we added to multi.py
data_path = "./multiwoz_data/data/MultiWOZ_2.2"
dataset = load_local_multiwoz(data_path)

# -------- 2. Define travel domains --------
TRAVEL_DOMAINS = {"hotel", "restaurant", "attraction", "taxi", "train", "bus"}

# -------- 3. Filter only travel dialogues --------
def is_travel_dialogue(dialogue):
    # 'services' field contains a list of domains for this dialogue in the raw JSON
    # Note: Hugging Face version used 'domains', raw uses 'services'
    if 'services' in dialogue:
        return bool(set(dialogue['services']) & TRAVEL_DOMAINS)
    return False

filtered_train = [d for d in dataset['train'] if is_travel_dialogue(d)]
filtered_dev   = [d for d in dataset['validation'] if is_travel_dialogue(d)]
filtered_test  = [d for d in dataset['test'] if is_travel_dialogue(d)]

# -------- 4. Convert to DataFrame for easy analysis --------
# When converting to DataFrame, we might want to flatten or select specific fields
# For now, just dumping the whole dict structure as before
train_df = pd.DataFrame(filtered_train)
dev_df   = pd.DataFrame(filtered_dev)
test_df  = pd.DataFrame(filtered_test)

# -------- 5. Save to CSV --------
train_df.to_csv("multiwoz_travel_train.csv", index=False)
dev_df.to_csv("multiwoz_travel_dev.csv", index=False)
test_df.to_csv("multiwoz_travel_test.csv", index=False)

print(f"Filtered train dialogues: {len(train_df)}")
print(f"Filtered dev dialogues: {len(dev_df)}")
print(f"Filtered test dialogues: {len(test_df)}")