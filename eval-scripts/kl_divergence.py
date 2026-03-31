import json
import glob
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt  

# ==============================
# CONFIG
# ==============================

GENERATED_DIALOGUES_FOLDER = "conversation_logs/"
MULTIWOZ_PATH = "ds-eval/cleaned.json"

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
    """
    Classify a given text into a dialogue act.

    Args:
        text (str): Input utterance.

    Returns:
        str: One of the dialogue acts: "REQUEST", "INFORM", "CONFIRM", "DENY", "GREET".
    """
    t = text.lower().strip()

    # GREET
    if any(x in t for x in ["hello", "hi", "thanks", "thank you", "bye", "goodbye", "you're welcome"]):
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
    """
    Load user utterances from generated dialogue JSON files.

    - Iterates over all JSON files in the folder.
    - Extracts utterances where speaker is "Model A".
    - Strips quotation marks and whitespace.
    - Skips malformed or empty turns.

    Args:
        folder_path (str): Folder containing generated dialogue JSON files.

    Returns:
        list[str]: List of cleaned user utterances.
    """
    user_texts = []

    for file_path in glob.glob(f"{folder_path}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversation = data.get("conversation", [])
        if not conversation:
            print(f" {file_path} empty or no conversation")
            continue

        for turn in conversation:
            if not isinstance(turn, list) or len(turn) < 2:
                print(f" Wrong turn: {turn}")
                continue

            speaker = str(turn[0]).strip()
            utterance = str(turn[1]).strip()

            # debug output
            print(f"DEBUG: speaker='{speaker}' utterance='{utterance[:30]}...'")

            if speaker.lower().replace(":", "") == "model a":
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
    """
    Load user utterances from MultiWOZ dataset, optionally filtering by allowed domains.

    Args:
        multiwoz_path (str): Path to MultiWOZ JSON file.
        allowed_domains (set[str] | None): Domains to include. If None, include all.

    Returns:
        list[str]: List of user utterances.
    """
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
                if len(text) > 0:
                    user_texts.append(text)

    return user_texts

# ==============================
# COMPUTE ACT DISTRIBUTION
# ==============================

def compute_act_distribution(texts):
    """
    Compute the probability distribution of dialogue acts in a list of texts.

    Args:
        texts (list[str]): List of utterances.

    Returns:
        dict[str, float]: Probability distribution of dialogue acts.
    """
    acts = [classify_dialogue_act(t) for t in texts]
    counts = Counter(acts)
    total = sum(counts.values())

    # Convert counts to probabilities, protect against division by zero
    dist = {act: counts.get(act, 0) / (total if total > 0 else 1) for act in DIALOGUE_ACTS}
    return dist

# ==============================
# KL DIVERGENCE
# ==============================

def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL Divergence D_KL(P || Q) for dialogue act distributions.

    Args:
        p (dict[str, float]): First distribution (e.g., generated).
        q (dict[str, float]): Second distribution (e.g., MultiWOZ reference).
        epsilon (float): Small value to avoid log(0).

    Returns:
        float: KL divergence value.
    """
    p_vals = np.array([p[a] for a in DIALOGUE_ACTS]) + epsilon
    q_vals = np.array([q[a] for a in DIALOGUE_ACTS]) + epsilon
    return np.sum(p_vals * np.log(p_vals / q_vals))

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Load user utterances
    gen_user_texts = load_generated_user_utterances(GENERATED_DIALOGUES_FOLDER)
    mwoz_user_texts = load_multiwoz_user_utterances(MULTIWOZ_PATH, allowed_domains=ALLOWED_DOMAINS)

    print(f"Generated user utterances: {len(gen_user_texts)}")
    print(f"MultiWOZ user utterances: {len(mwoz_user_texts)}")

    # Compute dialogue act distributions
    gen_dist = compute_act_distribution(gen_user_texts)
    mwoz_dist = compute_act_distribution(mwoz_user_texts)

    print("\nGenerated distribution:")
    print(gen_dist)
    print("\nMultiWOZ distribution:")
    print(mwoz_dist)

    # Compute KL Divergence
    kl = kl_divergence(gen_dist, mwoz_dist)
    kl_reverse = kl_divergence(mwoz_dist, gen_dist)

    print(f"\nKL Divergence (Generated || MultiWOZ): {kl:.4f}")
    print(f"KL Divergence (MultiWOZ || Generated): {kl_reverse:.4f}")

    # ==============================
    # PLOT DISTRIBUTION COMPARISON
    # ==============================

    x = np.arange(len(DIALOGUE_ACTS))
    width = 0.35

    plt.bar(x - width/2, [gen_dist[a] for a in DIALOGUE_ACTS], width, label="Generated (Model A)")
    plt.bar(x + width/2, [mwoz_dist[a] for a in DIALOGUE_ACTS], width, label="MultiWOZ USER")

    plt.xticks(x, DIALOGUE_ACTS)
    plt.ylabel("Frequency")
    plt.title("Dialogue Act Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()
