import json
import glob
import re
from collections import Counter
import numpy as np
from statistics import mean
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
GENERATED_DIALOGUES_FOLDER = "conversation_logs/"
MULTIWOZ_PATH = "ds-eval/cleaned.json"
ALLOWED_DOMAINS = {"hotel", "restaurant", "train"}

# ==============================
# TEXT PREPROCESSING
# ==============================
def tokenize(text):
    """
    Tokenize input text into lowercase word tokens.

    - Converts text to lowercase.
    - Removes non-alphanumeric characters.
    - Splits text by whitespace.

    Args:
        text (str): Input string.

    Returns:
        list[str]: List of tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def get_ngrams(tokens, n):
    """
    Generate n-grams from a list of tokens.

    Args:
        tokens (list[str]): List of word tokens.
        n (int): Size of n-grams.

    Returns:
        list[tuple]: List of n-gram tuples.
                     Returns empty list if len(tokens) < n.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ==============================
# LOAD DIALOGUES
# ==============================
def load_generated_dialogues(folder_path):
    """
    Load generated dialogues from JSON files.

    Args:
        folder_path (str): Folder containing JSON dialogue files.

    Returns:
        list[list]: List of dialogues, each dialogue is a list of turns.
    """
    dialogues = []
    for file_path in glob.glob(f"{folder_path}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        conv = data.get("conversation", [])
        if conv:
            dialogues.append(conv)
    return dialogues

def load_multiwoz_dialogues(path, allowed_domains=None):
    """
    Compute the average number of turns per dialogue.

    Args:
        dialogues (list[list]): List of dialogues.

    Returns:
        float: Average turns per dialogue, 0 if empty.
    """
    dialogues = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for dial in data:
        if allowed_domains and not any(s in allowed_domains for s in dial.get("services", [])):
            continue
        turns = dial.get("turns", [])
        if turns:
            dialogues.append(turns)
    return dialogues

# ==============================
# METRIC FUNCTIONS
# ==============================
def avg_turns_per_dialogue(dialogues):
    return mean(len(d) for d in dialogues) if dialogues else 0

def avg_words_per_user_utterance(dialogues, speaker="Model A"):
    """
    Compute the average number of words per utterance for a given speaker.

    Supports both list-style dialogues (generated) and dict-style (MultiWOZ).

    Args:
        dialogues (list): List of dialogues.
        speaker (str): Speaker to analyze.

    Returns:
        float: Average words per utterance, 0 if none found.
    """
    word_counts = []
    for dial in dialogues:
        # handle both list-style (generated) and dict-style (MultiWOZ)
        if isinstance(dial, list):
            turns = dial
        elif isinstance(dial, dict):
            turns = dial.get("turns", [])
        else:
            continue

        for turn in turns:
            if isinstance(turn, list) and len(turn) >= 2:
                spk = str(turn[0]).lower().replace(":", "")
                text = str(turn[1])
            elif isinstance(turn, dict):
                spk = str(turn.get("speaker", "")).lower().replace(":", "")
                text = str(turn.get("utterance", ""))
            else:
                continue

            if spk == speaker.lower().replace(":", ""):
                tokens = text.split()
                word_counts.append(len(tokens))

    return mean(word_counts) if word_counts else 0

def distinct_ngrams(dialogues, n=1, speaker="Model A"):
    """
    Compute the number of distinct n-grams for a speaker.

    Args:
        dialogues (list): List of dialogues.
        n (int): N-gram size.
        speaker (str): Speaker to analyze.

    Returns:
        int: Count of unique n-grams.
    """
    ngram_set = set()
    for dial in dialogues:
        if isinstance(dial, list):
            turns = dial
        elif isinstance(dial, dict):
            turns = dial.get("turns", [])
        else:
            continue

        for turn in turns:
            if isinstance(turn, list) and len(turn) >= 2:
                spk = str(turn[0]).lower().replace(":", "")
                text = str(turn[1])
            elif isinstance(turn, dict):
                spk = str(turn.get("speaker", "")).lower().replace(":", "")
                text = str(turn.get("utterance", ""))
            else:
                continue

            if spk == speaker.lower().replace(":", ""):
                tokens = tokenize(text)
                ngram_set.update(get_ngrams(tokens, n))
    return len(ngram_set)

def flatten_utterances(dialogues, speaker="Model A"):
    """
    Flatten all utterances for a given speaker into a single list.

    Args:
        dialogues (list): List of dialogues.
        speaker (str): Speaker to extract.

    Returns:
        list[str]: List of all utterances by the speaker.
    """
    texts = []
    for dial in dialogues:
        if isinstance(dial, list):
            turns = dial
        elif isinstance(dial, dict):
            turns = dial.get("turns", [])
        else:
            continue
        for turn in turns:
            if isinstance(turn, list) and len(turn) >= 2:
                spk = str(turn[0]).lower().replace(":", "")
                text = str(turn[1])
            elif isinstance(turn, dict):
                spk = str(turn.get("speaker", "")).lower().replace(":", "")
                text = str(turn.get("utterance", ""))
            else:
                continue
            if spk == speaker.lower().replace(":", ""):
                texts.append(text)
    return texts

def embedding_similarity(gen_texts, ref_texts, model_name="all-MiniLM-L6-v2"):
    """
    Compute semantic similarity between generated and reference texts using BERT embeddings.

    Args:
        gen_texts (list[str]): Generated texts.
        ref_texts (list[str]): Reference texts.
        model_name (str): SentenceTransformer model name.

    Returns:
        float: Mean cosine similarity between embeddings, 0 if any list is empty.
    """
    if not gen_texts or not ref_texts:
        return 0
    model = SentenceTransformer(model_name)
    gen_emb = model.encode(gen_texts, convert_to_tensor=True)
    ref_emb = model.encode(ref_texts, convert_to_tensor=True)
    similarity = util.cos_sim(gen_emb, ref_emb).mean().item()
    return similarity

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    gen_dialogues = load_generated_dialogues(GENERATED_DIALOGUES_FOLDER)
    mwoz_dialogues = load_multiwoz_dialogues(MULTIWOZ_PATH, allowed_domains=ALLOWED_DOMAINS)

    print(f"Generated dialogues: {len(gen_dialogues)}")
    print(f"MultiWOZ dialogues: {len(mwoz_dialogues)}")

    # Compute metrics
    metrics = {
        "Avg Turns per Dialogue": (
            avg_turns_per_dialogue(gen_dialogues),
            avg_turns_per_dialogue(mwoz_dialogues)
        ),
        "Avg Words per Utterance": (
            avg_words_per_user_utterance(gen_dialogues, speaker="Model A"),
            avg_words_per_user_utterance(mwoz_dialogues, speaker="USER")
        ),
        "Distinct-1": (
            distinct_ngrams(gen_dialogues, n=1, speaker="Model A"),
            distinct_ngrams(mwoz_dialogues, n=1, speaker="USER")
        ),
        "Distinct-2": (
            distinct_ngrams(gen_dialogues, n=2, speaker="Model A"),
            distinct_ngrams(mwoz_dialogues, n=2, speaker="USER")
        ),
    }

    # Embedding similarity
    gen_texts = flatten_utterances(gen_dialogues, speaker="Model A")
    mwoz_texts = flatten_utterances(mwoz_dialogues, speaker="USER")
    emb_sim = embedding_similarity(gen_texts, mwoz_texts)
    metrics["Semantic similarity (BERT)"] = (emb_sim, 1.0) 

    # Print metrics
    print("\nDialogue Metrics:")
    for name, (gen_val, mwoz_val) in metrics.items():
        print(f"{name}: Generated={gen_val:.4f}, MultiWOZ={mwoz_val:.4f}")

# ==============================
# PLOT EACH METRIC SEPARATELY
# ==============================
for name, (gen_val, mwoz_val) in metrics.items():
    plt.figure(figsize=(5, 5))
    plt.bar(["Generated", "MultiWOZ"], [gen_val, mwoz_val], color=["skyblue", "salmon"])
    plt.title(name)
    plt.ylabel("Value")
    plt.ylim(0, max(gen_val, mwoz_val) * 1.2) 
    for i, v in enumerate([gen_val, mwoz_val]):
        plt.text(i, v + 0.01 * max(gen_val, mwoz_val), f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()
