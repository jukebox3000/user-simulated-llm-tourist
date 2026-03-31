from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import json
import time
import os
import random
import streamlit as st
from spacy_ner_script import extract_entities


if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'ontology' not in st.session_state:
    st.session_state.ontology = {}
if 'persona' not in st.session_state:
    st.session_state.persona = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'is_complete' not in st.session_state:
    st.session_state.is_complete = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'pipelineA' not in st.session_state:
    st.session_state.pipelineA = None
if 'pipelineB' not in st.session_state:
    st.session_state.pipelineB = None
if 'current_turn' not in st.session_state:
    st.session_state.current_turn = 0
if 'total_turns' not in st.session_state:
    st.session_state.total_turns = 10
if 'current_key' not in st.session_state:
    st.session_state.current_key = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = {}


BASE_LLAMA_PATH = "/home/hpc/v132ca/v132ca25/llm-dialog-project/llama3.2-3b-4bit"
LORA_ADAPTER_PATH = "/home/hpc/v132ca/v132ca25/llm-dialog-project/travel-lora"
MODEL_B_PATH = "/home/hpc/v132ca/v132ca25/llm-dialog-project/gemma-2b-travel-qa/snapshots/d718695bf69d2f1a630d34703c1d578ca068ba32"
MAX_TURNS = 10


def apply_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-title {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        color: #1d1d1f;
        font-size: 28px;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #86868b;
        font-size: 14px;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e5e7;
    }
    
    .sidebar-header {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        color: #1d1d1f;
        font-size: 18px;
        margin-bottom: 1rem;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    }
    
    /* User message bubble (RIGHT side - iOS blue) */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 16px;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 4px 18px 18px;
        max-width: 70%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 15px;
        line-height: 1.4;
        box-shadow: 0 2px 5px rgba(0, 122, 255, 0.15);
        animation: fadeInUp 0.3s ease-out;
    }
    
    .user-bubble p {
        margin: 0;
    }
    
    /* Guide message bubble (LEFT side - iOS gray) */
    .guide-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 16px;
    }
    
    .guide-bubble {
        background-color: #f2f2f7;
        color: #1d1d1f;
        padding: 12px 16px;
        border-radius: 4px 18px 18px 18px;
        max-width: 70%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 15px;
        line-height: 1.4;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        animation: fadeInUp 0.3s ease-out;
    }
    
    .guide-bubble p {
        margin: 0;
    }
    
    /* Message sender label */
    .sender-label {
        font-size: 12px;
        color: #8e8e93;
        margin-bottom: 4px;
        font-weight: 500;
    }
    
    .user-label {
        text-align: right;
        padding-right: 10px;
    }
    
    .guide-label {
        text-align: left;
        padding-left: 10px;
    }
    
    /* Time indicator */
    .time-indicator {
        font-size: 11px;
        color: #c7c7cc;
        margin-top: 4px;
        text-align: right;
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Button styling */
    .stButton button {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 500;
        border-radius: 10px;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Status indicators */
    .status-box {
        background-color: #f2f2f7;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #007AFF;
    }
    
    .status-title {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        color: #1d1d1f;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .status-value {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #007AFF;
        font-size: 16px;
        font-weight: 600;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #c7c7cc;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8ad;
    }
    
    /* Progress indicator */
    .progress-indicator {
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 8px 12px;
        background-color: rgba(0, 122, 255, 0.1);
        border-radius: 10px;
        font-size: 13px;
        color: #007AFF;
    }
    
    .progress-dot {
        width: 8px;
        height: 8px;
        background-color: #007AFF;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 8px 16px;
        background-color: #f2f2f7;
        border-radius: 18px;
        width: fit-content;
        margin-left: 10px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        background-color: #8e8e93;
        border-radius: 50%;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-4px);
        }
    }
    </style>
    """, unsafe_allow_html=True)


def load_models():
    """Load models if not already loaded"""
    if st.session_state.models_loaded:
        return st.session_state.pipelineA, st.session_state.pipelineB
    
    with st.spinner("🔄 Loading models... This may take a minute..."):

        tokenizerA = AutoTokenizer.from_pretrained(BASE_LLAMA_PATH)
        modelA_base = AutoModelForCausalLM.from_pretrained(
            BASE_LLAMA_PATH,
            dtype=torch.float16,
            device_map="auto"
        )
        modelA = PeftModel.from_pretrained(modelA_base, LORA_ADAPTER_PATH)
        pipelineA = pipeline(
            "text-generation",
            model=modelA,
            tokenizer=tokenizerA,
            max_new_tokens=100,
            temperature=0.7
        )
        

        tokenizerB = AutoTokenizer.from_pretrained(MODEL_B_PATH)
        modelB = AutoModelForCausalLM.from_pretrained(
            MODEL_B_PATH,
            dtype=torch.float16,
            device_map="auto"
        )
        pipelineB = pipeline(
            "text-generation",
            model=modelB,
            tokenizer=tokenizerB,
            max_new_tokens=200,
            temperature=0.5
        )
        

        st.session_state.pipelineA = pipelineA
        st.session_state.pipelineB = pipelineB
        st.session_state.models_loaded = True
    
    return pipelineA, pipelineB


def chat(pipeline, prompt):
    """Generate response from a pipeline - EXACTLY from original code"""
    try:
        output = pipeline(prompt, return_full_text=False)[0]["generated_text"]
        return output.strip()
    except Exception as e:
        return f"Error: {e}"

def add_message(speaker, text):
    """Add a message to the conversation"""

    if speaker == "Model A" or speaker == "User":
        st.session_state.conversation.append({"speaker": "User", "text": text, "time": time.time()})
        st.session_state.conversation_history.append(("Model A", text))
    else:
        st.session_state.conversation.append({"speaker": "Guide", "text": text, "time": time.time()})
        st.session_state.conversation_history.append(("Model B", text))

def load_data():
    """Load personas, ontology, and dictionary - EXACTLY from original code"""
    try:

        with open("personas.json", "r") as f:
            personas = json.load(f)
        persona = random.choice(personas)
        

        with open("ontology_checklist.json", "r", encoding="utf-8") as f:
            full_ontology = json.load(f)
        k = random.randint(1, len(full_ontology))
        random_keys = random.sample(list(full_ontology.keys()), k)
        ontology = {key: full_ontology[key] for key in random_keys}
        
 
        with open("semantic_dictionary.json", "r", encoding="utf-8") as f:
            dictionary = json.load(f)
        
        print(f"Ontology: {ontology}\n")
        print(f"User Persona: {persona['name']}")
        
        return persona, ontology, dictionary
    except Exception as e:
        return None, {}, {}

def save_conversation():
    """Save conversation to file - EXACTLY from original code"""
    try:
        os.makedirs("conversation_logs", exist_ok=True)
        filename = f"conversation_logs/conv_{int(time.time())}.json"
        

        conv_for_save = []
        for msg in st.session_state.conversation_history:
            if msg[0] == "Model A":
                conv_for_save.append(("Model A", msg[1]))
            else:
                conv_for_save.append(("Model B", msg[1]))
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "persona": st.session_state.persona['name'] if st.session_state.persona else "Unknown",
                "conversation": conv_for_save,
                "ontology": st.session_state.ontology,
                "timestamp": time.time()
            }, f, indent=2)
        return filename
    except Exception as e:
        return None


def run_conversation_step():
    """Run one step of the conversation - using ORIGINAL LOGIC"""
    
    if not st.session_state.is_running:
        return
    

    pipelineA = st.session_state.pipelineA
    pipelineB = st.session_state.pipelineB
    persona = st.session_state.persona
    ontology = st.session_state.ontology
    current_turn = st.session_state.current_turn
    conversation_history = st.session_state.conversation_history
    
    print(f"DEBUG: Step {current_turn}, history length: {len(conversation_history)}")
    
   
    if current_turn == 0:
        if not persona:
            persona, ontology, dictionary = load_data()
            st.session_state.persona = persona
            st.session_state.ontology = ontology
            st.session_state.dictionary = dictionary
        

        initial_prompt = (
            f"You are a {persona['age']} {persona['name']}. "
            "Generate ONE realistic travel question, about a general first-impression question about ONE specific city in Europe"
            "(opinion and vibes, or what it's known for). "
            f"You prefer {', '.join(persona['preferences'])}. "
            "Use a curious, informal tone, max 30 words."
        )
        
        message_A = chat(pipelineA, initial_prompt)
        print(f"TURN 1 >>> Model A: {message_A}\n")
        add_message("Model A", message_A)
        

        TRAVEL_GUIDE_PROMPT = (
            "You are an expert travel guide.\n"
            f"The User asked: {message_A}\n"
            "Give only ONE answer to the user's travel question, concisely and specifically.\n"
            "Focus only on practical information about the city and travel plan.\n"
            "Use a formal and professional tone.\n"
            "Do not ask follow-up questions.\n"
            "Keep your answer under 120 words.\nAnswer:"
        )
        
        message_B = chat(pipelineB, TRAVEL_GUIDE_PROMPT)
        print(f"TURN 1 >>> Model B: {message_B}\n")
        add_message("Model B", message_B)
        
        st.session_state.current_turn = 1
        return
    

    ontology_turns = current_turn - 1
    null_keys = [key for key, value in ontology.items() if value.get("populated") is None]
    
    print(f"DEBUG: ontology_turns={ontology_turns}, null_keys={len(null_keys)}, MAX_TURNS={MAX_TURNS}")
    
    if ontology_turns >= (MAX_TURNS - 1) or not null_keys:

        print("DEBUG: Time for final messages")
        
 
        recent_conv = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        
   
        happy_final_promptA = (
            f"You are a {persona['age']} {persona['name']}.\n"
            f"Previous conversation: {recent_conv}\n"
            "Task: Write ONE informal sentence (under 30 words) expressing satisfaction with the travel planning conversation.\n"
            "Rules:\n"
            "- Output ONLY the sentence.\n"
            "- Do NOT explain the task.\n"
            "- Do NOT continue the dialogue.\n"
            "Sentence:"
        )
        
        dissatisfied_final_promptA = (
            f"You are a {persona['age']} {persona['name']}.\n"
            f"Previous conversation: {recent_conv}\n"
            "Task: Write ONE informal, annoyed sentence (under 30 words) expressing dissatisfaction.\n"
            "Rules:\n"
            "- Use simple, natural language.\n"
            "- Output ONLY the sentence.\n"
            "- Do NOT explain the task.\n"
            "- Do NOT continue the dialogue.\n"
            "Sentence:"
        )
        

        if current_turn > 8:
            print(f"DEBUG: Using DISSATISFIED final prompt (current_turn={current_turn})")
            message_A = chat(pipelineA, dissatisfied_final_promptA)
        else:
            print(f"DEBUG: Using HAPPY final prompt (current_turn={current_turn})")
            message_A = chat(pipelineA, happy_final_promptA)
        
        print(f"USER: {message_A}")
        add_message("Model A", message_A)
        conversation_history.append(("Model A", message_A))
        

        final_promptB = (
            "Respond to the traveler's message with ONE polite acknowledgment sentence."
            f"User said: {message_A}\n"
            "Your response:"
        )
        
        message_B = chat(pipelineB, final_promptB)
        print(f"GUIDE: {message_B}")
        add_message("Model B", message_B)
        conversation_history.append(("Model B", message_B))
        

        filename = save_conversation()
        if filename:
            print(f"✓ Conversation saved to: {filename}")
        

        print("\n" + "="*60)
        print("CONVERSATION COMPLETE")
        print("="*60)
        print(f"Final ontology state:")
        for key, value in ontology.items():
            populated = "✓" if value.get("populated") else "✗" if value.get("populated") is False else "?"
            entities_list = value.get('entities', [])
            print(f"  {populated} {key}: {entities_list}")
        print("="*60)
        
     
        st.session_state.is_complete = True
        st.session_state.is_running = False
        st.session_state.conversation_history = conversation_history
        return
    

    null_keys = [key for key, value in ontology.items() if value.get("populated") is None]
    if not null_keys:
        st.session_state.current_turn = MAX_TURNS
        return
    

    if st.session_state.current_key is None:
        selected_key = random.choice(null_keys)
        st.session_state.current_key = selected_key
    else:
        selected_key = st.session_state.current_key
    
    print(f"\n{'='*50}")
    print(f"TURN {current_turn} - Processing: {selected_key}")
    print(f"Ontology status: {ontology}")
    print(f"{'='*50}\n")
    

    recent_conv = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    

    next_promptA = (
        "You are a user-simulator.\n"
        f"Previous conversation: {recent_conv}\n"
        f"Ask ONLY ONE specific travel-related question about {selected_key}.\n"
        "Make it a single, short, clear question.\n"
        "Do NOT repeat previous questions.\n"
        "Question:"
    )
    
    message_A = chat(pipelineA, next_promptA)
    print(f"USER: {message_A}")
    add_message("Model A", message_A)
    conversation_history.append(("Model A", message_A))
    

    next_promptB = (
        "You are an expert travel guide.\n"
        f"Previous conversation context: {recent_conv}\n"
        "Answer the user's question clearly and concisely.\n"
        "Provide practical, specific information.\n"
        "Keep your answer under 100 words.\n"
        f"User question: {message_A}\n"
        "Answer:"
    )
    
    message_B = chat(pipelineB, next_promptB)
    print(f"GUIDE: {message_B}")
    add_message("Model B", message_B)
    conversation_history.append(("Model B", message_B))
    

    try:
        entities = extract_entities(message_B, selected_key)
        
        if entities:
            if len(entities) > 3:
                print(f"Too many entities found ({len(entities)}). Initiating follow-up question....")
                
                ontology[selected_key]["entities"] = entities
                ontology[selected_key]["populated"] = False
                
         
                followup_promptA = (
                    "You are a user-simulator.\n"
                    f"The travel guide gave too many options for {selected_key}: {entities}\n"
                    "Ask them to narrow it down to just ONE specific recommendation.\n"
                    "Make it a single polite request under 30 words.\n"
                    "Request:"
                )
                
                followup_A = chat(pipelineA, followup_promptA)
                print(f"FOLLOW-UP USER: {followup_A}")
                add_message("Model A", followup_A)
                conversation_history.append(("Model A", followup_A))
                
               
                followup_promptB = (
                    "You are an expert travel guide.\n"
                    f"The user wants you to narrow down the options for {selected_key}.\n"
                    "Choose ONE specific, best recommendation from your previous answer.\n"
                    "Explain briefly why you chose it.\n"
                    f"Previous options: {entities}\n"
                    "Recommendation:"
                )
                
                followup_B = chat(pipelineB, followup_promptB)
                print(f"FOLLOW-UP GUIDE: {followup_B}")
                add_message("Model B", followup_B)
                conversation_history.append(("Model B", followup_B))
                
                
                followup_entities = extract_entities(followup_B, selected_key)
                if followup_entities:
                    ontology[selected_key]["entities"] = followup_entities[:1]
                    ontology[selected_key]["populated"] = True
                    st.session_state.current_key = None
                    print(f"✓ {selected_key} populated with: {followup_entities[:1]}")
                else:
                    print(f"✗ No entities in follow-up. Keeping {selected_key} unpopulated.")
                    ontology[selected_key]["populated"] = False
                    st.session_state.current_key = selected_key
            
            else:
                ontology[selected_key]["entities"] = entities
                ontology[selected_key]["populated"] = True
                st.session_state.current_key = None
                print(f"✓ {selected_key} populated with: {entities}")
        
        else:
            print(f"✗ No entities found for {selected_key}.")
            ontology[selected_key]["populated"] = False
            st.session_state.current_key = selected_key
    
    except Exception as e:
        print(f"Error in ontology population: {e}")
        ontology[selected_key]["populated"] = False
        st.session_state.current_key = selected_key
    
 
    print(f"\nCurrent ontology: {ontology}")
    print(f"{'='*50}\n")
    
    
    st.session_state.ontology = ontology
    st.session_state.conversation_history = conversation_history
    st.session_state.current_turn += 1


def main():

    apply_custom_css()
    
    st.set_page_config(
        page_title="LLM Dialogue Simulator",
        page_icon="💬",
        layout="wide"
    )
    
  
    st.markdown('<h1 class="main-title">💬 Travel Assistant Chat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Real-time conversation between User and Travel Guide</p>', unsafe_allow_html=True)
    
   
    col1, col2 = st.columns([1, 2])
    
    with col1:
      
        with st.container():
            st.markdown('<div class="sidebar-header">Controls</div>', unsafe_allow_html=True)
            
        
            if not st.session_state.is_running:
                if st.button("🚀 Start Chat", type="primary", use_container_width=True):
                    print("\n" + "="*60)
                    print("Running multi-turn conversation between two local LLMs...")
                    print("="*60 + "\n")
                    
                
                    st.session_state.conversation = []
                    st.session_state.conversation_history = []
                    st.session_state.ontology = {}
                    st.session_state.persona = None
                    st.session_state.current_turn = 0
                    st.session_state.current_key = None
                    st.session_state.is_running = True
                    st.session_state.is_complete = False
                    
                  
                    load_models()
                    
               
                    st.rerun()
            
          
            if st.session_state.is_running:
                if st.button("⏹️ Stop Chat", use_container_width=True):
                    st.session_state.is_running = False
                    st.session_state.is_complete = False
                    st.rerun()
            
          
            st.markdown('<div class="status-box">', unsafe_allow_html=True)
            st.markdown('<div class="status-title">Current Status</div>', unsafe_allow_html=True)
            
            if st.session_state.persona:
                st.markdown(f'<div class="status-value">👤 {st.session_state.persona["name"]}</div>', unsafe_allow_html=True)
            
            if st.session_state.is_running:
                if st.session_state.is_complete:
                    st.markdown('<div class="status-value">✅ Complete</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-value">🔄 Turn {st.session_state.current_turn}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-value">⏸️ Ready</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
          
            if st.session_state.persona:
                st.markdown('<div class="status-box">', unsafe_allow_html=True)
                st.markdown('<div class="status-title">Persona Details</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 13px; color: #666; margin-bottom: 5px;">Age: {st.session_state.persona["age"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 13px; color: #666;">Preferences: {", ".join(st.session_state.persona["preferences"])}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
      
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if not st.session_state.conversation:
         
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 48px; margin-bottom: 20px; color: #e5e5e7;">💬</div>
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #8e8e93; font-size: 16px; margin-bottom: 30px;">
                    Click "Start Chat" to begin a conversation
                </div>
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #c7c7cc; font-size: 14px;">
                    The AI will simulate a travel planning dialogue
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.conversation:
                if msg["speaker"] == "User":
                  
                    st.markdown(f'''
                    <div class="user-message">
                        <div style="flex: 1; max-width: 70%;">
                            <div class="sender-label user-label">You</div>
                            <div class="user-bubble">
                                {msg["text"]}
                            </div>
                            <div class="time-indicator">
                                {time.strftime('%I:%M %p', time.localtime(msg["time"]))}
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="guide-message">
                        <div style="flex: 1; max-width: 70%;">
                            <div class="sender-label guide-label">Travel Guide</div>
                            <div class="guide-bubble">
                                {msg["text"]}
                            </div>
                            <div class="time-indicator">
                                {time.strftime('%I:%M %p', time.localtime(msg["time"]))}
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            

            if st.session_state.is_running and not st.session_state.is_complete:
                st.markdown('''
                <div class="guide-message">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            

            if st.session_state.is_complete:
                st.markdown('''
                <div style="text-align: center; margin-top: 30px; padding: 15px; background: linear-gradient(135deg, #34c75920, #32d74b20); border-radius: 12px; border: 1px solid #34c75930;">
                    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 14px; color: #34c759; font-weight: 500;">
                        ✅ Conversation complete! Check terminal for details.
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        

        st.markdown(
            """
            <script>
            // Auto-scroll to bottom of chat
            function scrollToBottom() {
                var chatContainer = document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
            
            // Scroll on page load
            window.onload = scrollToBottom;
            
            // Create a MutationObserver to detect new messages
            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes.length) {
                        setTimeout(scrollToBottom, 100);
                    }
                });
            });
            
            // Start observing the chat container
            var chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                observer.observe(chatContainer, { childList: true, subtree: true });
            }
            </script>
            """,
            unsafe_allow_html=True
        )
    

    if st.session_state.is_running and not st.session_state.is_complete:
     
        time.sleep(0.5)
        run_conversation_step()
        st.rerun()


if __name__ == "__main__":
    main()