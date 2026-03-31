import json
import glob
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONFIG
# ==============================

GENERATED_DIALOGUES_FOLDER = "conversation_logs/"
MULTIWOZ_PATH = "ds-eval/cleaned.json"

# Filter MultiWOZ by domain
ALLOWED_DOMAINS = {"hotel", "restaurant", "train"}

DIALOGUE_ACTS = [
    "REQUEST",
    "INFORM",
    "CONFIRM",
    "DENY",
    "GREET"
]

# ==============================
# DIALOGUE ACT CLASSIFIER
# ==============================

def classify_dialogue_act(text: str) -> str:
    t = text.lower().strip()

    # GREET / THANK / BYE
    if any(x in t for x in [
        "hello", "hi", "thanks", "thank you",
        "bye", "goodbye", "you're welcome"
    ]):
        return "GREET"

    # CONFIRM
    if re.fullmatch(r"(yes|yeah|yep|correct|right|ok|okay|sounds good)", t):
        return "CONFIRM"

    # DENY
    if re.fullmatch(r"(no|nope|not really|actually no)", t):
        return "DENY"

    # REQUEST
    if "?" in t or any(x in t for x in [
        "can you", "could you", "do you have",
        "i want", "i need", "recommend",
        "looking for"
    ]):
        return "REQUEST"

    # INFORM (default)
    return "INFORM"

# ==============================
# LOAD GENERATED USER UTTERANCES
# ==============================

def load_generated_user_utterances(folder_path):
    user_texts = []

    for file_path in glob.glob(f"{folder_path}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for turn in data.get("conversation", []):

            # safety check
            if not isinstance(turn, list) or len(turn) < 2:
                continue

            speaker = str(turn[0]).strip().lower()
            utterance = str(turn[1]).strip()

            # normalize speaker
            if speaker.replace(":", "") == "model a":

                # remove wrapping quotes if they exist
                if utterance.startswith('"') and utterance.endswith('"'):
                    utterance = utterance[1:-1]

                utterance = utterance.strip()

                if utterance:
                    user_texts.append(utterance)

    return user_texts

# ==============================
# LOAD MULTIWOZ USER UTTERANCES
# ==============================

def load_multiwoz_user_utterances(multiwoz_path, allowed_domains=None):
    user_texts = []

    with open(multiwoz_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    for dial in dialogues:
        if allowed_domains is not None:
            if not any(s in allowed_domains for s in dial.get("services", [])):
                continue

        for turn in dial.get("turns", []):
            if turn.get("speaker") == "USER":
                text = turn.get("utterance", "").strip()
                if text:
                    user_texts.append(text)

    return user_texts

# ==============================
# COMPUTE ACT DISTRIBUTION
# ==============================

def compute_act_distribution(texts):
    """
    Returns:
        counts: Counter of dialogue acts
        dist: normalized distribution
    """
    if len(texts) == 0:
        return Counter(), {act: 0.0 for act in DIALOGUE_ACTS}

    acts = [classify_dialogue_act(t) for t in texts]
    counts = Counter(acts)

    total = sum(counts.values())
    dist = {act: counts.get(act, 0) / total for act in DIALOGUE_ACTS}

    return counts, dist

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Load data
    gen_user_texts = load_generated_user_utterances(GENERATED_DIALOGUES_FOLDER)
    mwoz_user_texts = load_multiwoz_user_utterances(
        MULTIWOZ_PATH,
        allowed_domains=ALLOWED_DOMAINS
    )

    print(f"Generated user utterances: {len(gen_user_texts)}")
    print(f"MultiWOZ user utterances: {len(mwoz_user_texts)}")

    # Compute distributions
    gen_counts, gen_dist = compute_act_distribution(gen_user_texts)
    mwoz_counts, mwoz_dist = compute_act_distribution(mwoz_user_texts)

    # ==============================
    # TABLE FOR REPORT
    # ==============================

    df = pd.DataFrame({
        "Dialogue Act": DIALOGUE_ACTS,
        "Generated (Model A)": [gen_dist[a] for a in DIALOGUE_ACTS],
        "MultiWOZ USER": [mwoz_dist[a] for a in DIALOGUE_ACTS]
    })

    print("\nDialogue Act Frequency Comparison:\n")
    print(df.to_string(index=False))

    # ==============================
    # PLOT
    # ==============================

    x = np.arange(len(DIALOGUE_ACTS))
    width = 0.35

    plt.bar(
        x - width / 2,
        [gen_dist[a] for a in DIALOGUE_ACTS],
        width,
        label="Generated (Model A)"
    )

    plt.bar(
        x + width / 2,
        [mwoz_dist[a] for a in DIALOGUE_ACTS],
        width,
        label="MultiWOZ USER"
    )

    plt.xticks(x, DIALOGUE_ACTS)
    plt.ylabel("Frequency")
    plt.title("Dialogue Act Distribution (User Behaviour)")
    plt.legend()
    plt.tight_layout()
    plt.show()
