import json
import glob
import re
from collections import Counter
import numpy as np
from itertools import islice

# ==============================
# CONFIG
# ==============================

GENERATED_DIALOGUES_FOLDER = "conversation_logs/"
MULTIWOZ_PATH = "ds-eval/cleaned.json"

ALLOWED_DOMAINS = {"hotel", "restaurant", "train"}

N_GRAM = 2  # Can be changed according to n value

# ==============================
# TEXT PREPROCESSING
# ==============================

def tokenize(text):
    """
    Tokenize input text into lowercase word tokens.

    This function:
    - Converts text to lowercase.
    - Removes non-alphanumeric characters.
    - Splits text by whitespace.

    Args:
        text (str): Input text string.

    Returns:
        list[str]: List of cleaned tokens.
    """
    text = text.lower()
    # Simple whitespace tokenization with punctuation removal
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens

def get_ngrams(tokens, n):

    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ==============================
# LOAD GENERATED USER UTTERANCES
# ==============================

def load_generated_user_utterances(folder_path):
    """
    Generate n-grams from a list of tokens.

    Args:
        tokens (list[str]): List of word tokens.
        n (int): Size of n-grams.

    Returns:
        list[tuple]: List of n-gram tuples.
        Returns an empty list if len(tokens) < n.
    """
    user_texts = []

    for file_path in glob.glob(f"{folder_path}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for turn in data.get("conversation", []):
            if not isinstance(turn, list) or len(turn) < 2:
                continue
            speaker = str(turn[0]).strip().lower()
            utterance = str(turn[1]).strip()
            if speaker.replace(":", "") == "model a":
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
    Load USER utterances from the MultiWOZ dataset.

    Optionally filters dialogues by allowed service domains.

    Args:
        multiwoz_path (str): Path to cleaned MultiWOZ JSON file.
        allowed_domains (set[str] | None):
            If provided, only dialogues containing at least one
            of these domains are included.

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
                if text:
                    user_texts.append(text)

    return user_texts

# ==============================
# BUILD N-GRAM DISTRIBUTION
# ==============================

def ngram_distribution(texts, n):
    """
    Build a normalized n-gram probability distribution.

    The function:
    - Tokenizes each text.
    - Extracts n-grams.
    - Counts frequencies using Counter.
    - Converts counts to probabilities.

    Args:
        texts (list[str]): List of text strings.
        n (int): N-gram size.

    Returns:
        dict[tuple, float]: Dictionary mapping n-grams to probabilities.
        Returns empty dict if no n-grams are found.
    """
    counter = Counter()
    for t in texts:
        tokens = tokenize(t)
        ngrams = get_ngrams(tokens, n)
        counter.update(ngrams)

    total = sum(counter.values())

    dist = {k: v / total for k, v in counter.items()} if total > 0 else {}
    return dist

# ==============================
# KL DIVERGENCE
# ==============================

def kl_divergence(p, q, epsilon=1e-10):
    """
   Compute KL Divergence D_KL(P || Q).

    KL divergence measures how one probability distribution P
    diverges from another distribution Q.

    Smoothing with epsilon is applied to avoid log(0).

    Args:
        p (dict): First probability distribution.
        q (dict): Second probability distribution.
        epsilon (float): Small smoothing constant.

    Returns:
        float: KL divergence value.
    """
    keys = set(p.keys()).union(q.keys())
    p_vals = np.array([p.get(k, 0) + epsilon for k in keys])
    q_vals = np.array([q.get(k, 0) + epsilon for k in keys])
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

    # Build n-gram distributions
    gen_dist = ngram_distribution(gen_user_texts, N_GRAM)
    mwoz_dist = ngram_distribution(mwoz_user_texts, N_GRAM)

    print(f"\nGenerated {N_GRAM}-gram distribution size: {len(gen_dist)}")
    print(f"MultiWOZ {N_GRAM}-gram distribution size: {len(mwoz_dist)}")

    # Compute KL divergence
    kl = kl_divergence(gen_dist, mwoz_dist)
    kl_reverse = kl_divergence(mwoz_dist, gen_dist)

    print(f"\nKL Divergence (Generated || MultiWOZ): {kl:.4f}")
    print(f"KL Divergence (MultiWOZ || Generated): {kl_reverse:.4f}")

import matplotlib.pyplot as plt

# ==============================
# PLOT N-GRAM DISTRIBUTIONS
# ==============================
def plot_top_ngrams(gen_dist, mwoz_dist, top_k=20, title=f"Top {N_GRAM}-grams Comparison"):
    """
    Plot top-K n-grams comparing two distributions.

    The function:
    - Selects top n-grams based on maximum probability
      across both distributions.
    - Displays a side-by-side bar chart.

    Args:
        gen_dist (dict): Generated dialogue n-gram distribution.
        mwoz_dist (dict): MultiWOZ n-gram distribution.
        top_k (int): Number of top n-grams to display.
        title (str | None): Plot title.
    """

    all_ngrams = set(list(gen_dist.keys()) + list(mwoz_dist.keys()))
    top_ngrams = sorted(all_ngrams, key=lambda x: max(gen_dist.get(x,0), mwoz_dist.get(x,0)), reverse=True)[:top_k]

    x = range(len(top_ngrams))
    gen_vals = [gen_dist.get(ng,0) for ng in top_ngrams]
    mwoz_vals = [mwoz_dist.get(ng,0) for ng in top_ngrams]

    plt.figure(figsize=(12,6))
    width = 0.35
    plt.bar([i - width/2 for i in x], gen_vals, width, label="Generated (Model A)")
    plt.bar([i + width/2 for i in x], mwoz_vals, width, label="MultiWOZ USER")
    plt.xticks(x, [" ".join(ng) for ng in top_ngrams], rotation=90)
    plt.ylabel("Probability / Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==============================
# Plot distributions
# ==============================
plot_top_ngrams(gen_dist, mwoz_dist, top_k=20)

