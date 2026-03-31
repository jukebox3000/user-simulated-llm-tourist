import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from sentence_transformers import SentenceTransformer, util
import textstat
import spacy

# ==============================
# CONFIG
# ==============================
nlp = spacy.load("en_core_web_sm")
GENERATED_DIALOGUES_FOLDER = "conversation_logs/"
MULTIWOZ_PATH = "ds-eval/cleaned.json"
ALLOWED_DOMAINS = {"hotel", "restaurant", "train"}

# ==============================
# LOAD DIALOGUES
# ==============================
def load_generated_dialogues(folder_path):
    """
    Load generated dialogues from JSON files in a specified folder.

    - Iterates over all JSON files in the given folder.
    - Extracts the "conversation" field from each file.
    - Only includes non-empty conversations.
    - Returns a list of dialogues, each dialogue is a list of turns.

    Args:
        folder_path (str): Path to the folder containing generated dialogue JSON files.

    Returns:
        list[list]: List of dialogues. Each dialogue is a list of turns.
    """
    dialogues = []
    for file_path in glob.glob(f"{folder_path}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        conv = data.get("conversation", [])
        if conv:
            dialogues.append(conv)
    return dialogues

def load_multiwoz_dialogues(multiwoz_path, allowed_domains=None):
    """
    Load dialogues from the MultiWOZ dataset, optionally filtering by allowed domains.

    - Reads MultiWOZ JSON file.
    - If allowed_domains is provided, only dialogues containing at least one of these domains are included.
    - Returns a list of dialogues, each represented as a list of turns.

    Args:
        multiwoz_path (str): Path to the MultiWOZ JSON file.
        allowed_domains (set[str] | None): Set of domains to filter dialogues. If None, all dialogues are included.

    Returns:
        list[list]: List of dialogues, each as a list of turns.
    """
    dialogues = []
    with open(multiwoz_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for dial in data:
        if allowed_domains and not any(s in allowed_domains for s in dial.get("services", [])):
            continue
        dialogues.append(dial.get("turns", []))
    return dialogues

# ==============================
# HELPER TO EXTRACT UTTERANCES
# ==============================
def extract_texts(dialogues, speaker=None):
    """
    Extract all utterances from dialogues for a specific speaker (or all speakers if None).

    Args:
        dialogues (list): List of dialogues. Each dialogue can be a list (generated) 
                          or dict with a "turns" key (MultiWOZ).
        speaker (str | None): Speaker to filter by (case-insensitive). If None, include all speakers.

    Returns:
        list[str]: List of extracted utterances matching the speaker filter.
    """
    texts = []
    for dial in dialogues:
        turns = dial if isinstance(dial, list) else dial.get("turns", [])
        for turn in turns:
            # Generated dialogue: list-style
            if isinstance(turn, list) and len(turn) >= 2:
                spk = str(turn[0]).lower().replace(":", "")
                text = str(turn[1]).strip()
            # MultiWOZ dialogue: dict-style
            elif isinstance(turn, dict):
                spk = str(turn.get("speaker", "")).lower().replace(":", "")
                text = str(turn.get("utterance", "")).strip()
            else:
                continue
            if speaker is None or spk == speaker.lower():
                if text:
                    texts.append(text)
    return texts

# ==============================
# DOMAIN COVERAGE
# ==============================
def domain_coverage(dialogues, domains, speaker="Model A"):
    """
    Compute domain coverage for a given speaker within dialogue sessions.

    Coverage is calculated as:

        number_of_covered_domains / total_number_of_requested_domains

    Args:
        dialogues (list): A list of dialogue sessions containing
            speaker-labeled utterances.
        domains (list): A list of domain identifiers to evaluate
            (e.g., ["hotel", "restaurant", "train"]).
        speaker (str, optional): The speaker whose utterances should
            be analyzed. Defaults to "Model A".

    Returns:
        float: A normalized coverage score between 0 and 1.
            - 1.0 means all requested domains are covered.
            - 0.0 means none of the requested domains are covered.
            Returns 0 if `domains` is empty.
    """

    texts = extract_texts(dialogues, speaker)
    covered = set()
    
    # Keywords
    domain_categories = {
        "hotel": {
            "name": "accommodation",
            "keywords": [
                # Accomodation
                "hotel", "motel", "hostel", "inn", "lodge", 
                "accommodation", "apartment", "suite", "room", "studio",
                "airbnb", "bed and breakfast", "b&b", "vacation rental",
                "guest house", "villa", "chalet", "resort",
                # Booking context
                "booking", "reservation", "check-in", "checkout", "reception",
                "single room", "double room", "twin room", "family room",
                "deluxe room", "standard room", "executive suite",
                # Услуги
                "breakfast included", "free wifi", "parking", "pool",
                "air conditioning", "minibar", "room service", "spa",
                "gym", "fitness center", "laundry", "concierge"
            ]
        },
        "restaurant": {
            "name": "food",
            "keywords": [
                # Food places
                "restaurant", "cafe", "coffee shop", "bistro", "diner",
                "eatery", "pub", "tavern", "bar", "grill", "steakhouse",
                "pizzeria", "bakery", "food truck", "food court",
                "buffet", "cafeteria", "brasserie", "trattoria",
                # Food
                "food", "meal", "dining", "eating", "cuisine",
                "breakfast", "lunch", "dinner", "supper", "brunch",
                "snack", "appetizer", "main course", "entree", "dessert",
                # Context
                "menu", "dish", "course", "portion", "serving",
                "reservation", "booking", "table for", "dinner table",
                "takeaway", "takeout", "delivery", "catering",
                # Cuisines
                "italian", "chinese", "indian", "mexican", "japanese",
                "thai", "french", "greek", "spanish", "mediterranean",
                "vegetarian", "vegan", "gluten-free", "seafood", "bbq",
                # Beverage
                "drink", "beverage", "cocktail", "wine", "beer", "coffee"
            ]
        },
        "train": {
            "name": "transport",
            "keywords": [
                # Transport
                "train", "railway", "railroad", "metro", "subway", "underground",
                "bus", "coach", "shuttle", "minibus", "tram", "streetcar",
                "taxi", "cab", "rideshare", "uber", "lyft", "car service",
                "airplane", "flight", "airline", "airport", "aeroplane",
                "ferry", "boat", "ship", "yacht", "cruise", "water taxi",
                "bicycle", "bike", "scooter", "motorcycle", "motorbike",
                # Transportation
                "transport", "transportation", "transit", "commute", "travel",
                "public transport", "mass transit", "commuter", "passenger",
                # Booking ticket context
                "ticket", "booking", "reservation", "fare", "pass",
                "seat", "compartment", "berth", "cabin", "luggage",
                "departure", "arrival", "schedule", "timetable", "itinerary",
                "station", "platform", "terminal", "bus stop", "depot",
                "first class", "second class", "business class", "economy",
                "one-way", "round trip", "return ticket", "season ticket"
            ]
        }
    }
    
    for text in texts:
        text_lower = text.lower()
        
        for domain in domains:
            if domain not in domain_categories:
                continue
                
            category_info = domain_categories[domain]
            found = False
            
            # Checking keywords
            for keyword in category_info["keywords"]:
                if keyword in text_lower:
                    # Filter false positives
                    if keyword == "train" and any(fp in text_lower for fp in ["training", "trainer", "trained"]):
                        continue
                    if keyword == "food" and "food for thought" in text_lower:
                        continue
                    if keyword == "bar" and any(ctx in text_lower for ctx in ["bar exam", "bar association", "chocolate bar"]):
                        continue
                    
                    found = True
                    break
            
            if found:
                covered.add(domain)
    
    return len(covered) / len(domains) if domains else 0

# ==============================
# FLUENCY / FLESCH READING EASE
# ==============================
def avg_flesch_kincaid(dialogues, speaker="Model A"):
    """
    Compute the average Flesch Reading Ease score for a given speaker
    across multiple dialogue sessions.

    Args:
        dialogues (list): A list of dialogue sessions. Each session
            should contain speaker-labeled utterances.
        speaker (str, optional): The speaker whose texts should be
            analyzed. Defaults to "Model A".

    Returns:
        float: The average Flesch Reading Ease score across all valid
        texts of the specified speaker. Returns 0 if no valid scores
        are available.
    """
    texts = extract_texts(dialogues, speaker)
    scores = []
    for text in texts:
        try:
            score = textstat.flesch_reading_ease(text)
            scores.append(score)
        except:
            continue
    return mean(scores) if scores else 0

# ==============================
# SEMANTIC COHERENCE
# ==============================
def semantic_coherence(dialogues, speaker="Model A"):
    """
    Calculate semantic coherence by comparing consecutive utterances 
    of the specified speaker within dialogues.    
    Coherence is measured as the cosine similarity between embeddings
    of consecutive utterances from the same speaker.
    
    Args:
        dialogues: List of dialogue sessions
        speaker: Speaker ID to analyze (default: "Model A")
    
    Returns:
        float: Average cosine similarity between consecutive utterances
               Returns 0 if fewer than 2 utterances are found
    """
    # Extract only utterances from the specified speaker
    texts = extract_texts(dialogues, speaker)
    
    # Need at least 2 utterances to calculate coherence
    if len(texts) < 2:
        return 0.0
    
    # Initialize SentenceTransformer model for embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate embeddings for all utterances at once (more efficient)
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Calculate pairwise similarities between consecutive utterances
    similarities = []
    for i in range(len(embeddings) - 1):
        # Cosine similarity between utterance i and i+1
        sim = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
        similarities.append(sim)
    
    # Return average similarity, or 0 if no similarities were calculated
    return mean(similarities) if similarities else 0.0

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # Load dialogues
    gen_dialogues = load_generated_dialogues(GENERATED_DIALOGUES_FOLDER)
    mwoz_dialogues = load_multiwoz_dialogues(MULTIWOZ_PATH, allowed_domains=ALLOWED_DOMAINS)

    print(f"Generated dialogues: {len(gen_dialogues)}")
    print(f"MultiWOZ dialogues: {len(mwoz_dialogues)}")

    # Compute metrics
    metrics = {
        "Domain Coverage": (
            domain_coverage(gen_dialogues, ALLOWED_DOMAINS),
            domain_coverage(mwoz_dialogues, ALLOWED_DOMAINS, speaker="USER")
        ),
        "Fluency (Flesch Reading Ease)": (
            avg_flesch_kincaid(gen_dialogues),
            avg_flesch_kincaid(mwoz_dialogues, speaker="USER")
        ),
        "Semantic Coherence": (
            semantic_coherence(gen_dialogues),
            semantic_coherence(mwoz_dialogues, speaker="USER")
        )
    }

    # Print metrics
    print("\nQualitative Dialogue Metrics:")
    for name, (gen_val, mwoz_val) in metrics.items():
        print(f"{name}: Generated={gen_val:.4f}, MultiWOZ={mwoz_val:.4f}")

    # ==============================
    # PLOT EACH METRIC SEPARATELY
    # ==============================
    for name, (gen_val, mwoz_val) in metrics.items():
        plt.figure(figsize=(4,5))
        plt.bar(["Generated", "MultiWOZ"], [gen_val, mwoz_val], color=["skyblue", "orange"])
        plt.title(name)
        plt.ylabel("Score")
        plt.ylim(0, max(gen_val, mwoz_val)*1.2 if max(gen_val, mwoz_val)>0 else 1)
        for i, v in enumerate([gen_val, mwoz_val]):
            plt.text(i, v + 0.02*max(gen_val, mwoz_val), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.show()
