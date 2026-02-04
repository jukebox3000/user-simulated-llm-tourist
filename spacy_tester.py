import json
import spacy
nlp = spacy.load("en_core_web_sm")

# Load your semantic dictionary and topics
with open("semantic_dictionary.json", "r", encoding="utf-8") as f:
    semantic_dict = json.load(f)
with open("ontology_checklist.json", "r", encoding="utf-8") as f:
    ontology = json.load(f)

transport_keywords = semantic_dict.get("transport", {}).get("keywords", [])
#food_keywords = semantic_dict.get("food", {}).get("keywords", [])



def extract_entities(text, topic):
    transport_entities = []
#    food_entities = []
    entities = []
    words = text.lower().split()

    if topic.lower() == "transport":
        for word in words:
            if word.lower() in map(str.lower, transport_keywords):
                if word.lower() not in transport_entities:
                    transport_entities.append(word.lower())
        print("Transport entities:", transport_entities)

#    elif topic.lower() == "food":
#        for word in words:
#            if word.lower() in map(str.lower, food_keywords):
#                if word.lower() not in food_entities:
#                    food_entities.append(word.lower())
#        print("food_entities:", food_entities)

    else:
        # NORMAL NER logic
        topic_label_map = {
            "accommodation": ["ORG"],
            "food": ["ORG"],
            "sightseeing": ["LOC", "FAC"]
        }
        relevant_labels = topic_label_map.get(topic.lower(), [])
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in relevant_labels:
                entities.append(ent.text)
        print(f"{topic} entities:", entities)

    if topic.lower() == "transport":
        return transport_entities
#    elif topic.lower() == "food":
#        return food_entities
    else:
        return entities
    
