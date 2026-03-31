# User Simulation with Large Language Models for Tourism Dialogue Synthesis



## Overview

This project implements a dialog-based tourism application built using large language models (LLMs) and structured semantic processing. The system simulates user personas and generates tourism-related interactions using a two-model pipeline combined with entity recognition and ontology-based validation.

The project focuses on:
	•	LLM-driven dialog generation
	•	User simulation with predefined personas
	•	Entity recognition using spaCy
	•	Ontology-based validation and semantic dictionary matching
	•	Automated interaction flow through a structured pipeline

⸻

## Model Setup

This system uses:
    •	**User-simulator Model**: [*unsloth/Llama-3.2-3B-Instruct-bnb-4bit*](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit) with LoRA [*harishjan/llama3.2._travel-conversations_trained*](https://huggingface.co/harishjan/llama3.2._travel-conversations_trained)
    •	**Travel Guide Model**: [*ashmib/gemma-2b-travel-qa*](https://huggingface.co/ashmib/gemma-2b-travel-qa)

Both models were downloaded and executed locally.
Due to their size, the model weights are not included in this repository. To run this project:
	1.	Download the required models from their official documentation pages.
	2.	Place them in your local environment according to your inference framework setup.
	3.	Update the model paths inside the code if necessary.

You may also substitute these models with any other compatible LLM of your choice, provided the interface supports the same generation pipeline.

⸻

## Repository Structure

### simulator-script.ipynb
Main notebook containing the integrated pipeline implementation, testing workflow, and experimentation results.

### finalstreamlitui.py
Visual implementation of core application logic UI using Streamlit libraries.

### spacy_tester.py
Script for spaCy-based entity recognition and preprocessing validation.

### personas.json
Defines simulated user personas including demographic and preference-based attributes used during dialog generation.

### ontology_checklist.json
Structured ontology constraints used to validate tourism-related entities and conversation outputs.

### semantic_dictionary.json
Dictionary mapping domain-specific terminology and semantic categories for entity consistency checks.

⸻

## Dependencies
	•	Python 3.x
	•	spaCy
	•	LLM inference framework compatible with <MODEL_NAME_1> and <MODEL_NAME_2>

## Running the Code
> **Note on models:** During development, the LLM weights were downloaded locally while we ran experiments on a remote HPC cluster for access to large memory and GPUs. The repository does not include these weights. You can replace the model paths in the code with your own local or remote locations; the pipeline will work equally well if you point it to models stored on network drives or cloud storage and execute on a machine with sufficient resources.

1. **Basic pipeline execution (notebook)**
   - Open the `simulator-script.ipynb` notebook in Jupyter/VS Code and run the cells sequentially.
   - The notebook loads `semantic_dictionary.json`, `personas.json` and `ontology_checklist.json` and demonstrates the full dialog pipeline, including simulation, entity recognition and topic completion.
   - This is the quickest way to verify the core functionality without installing a UI.

2. **Streamlit application**
   - Ensure all dependencies are installed (see above) and that your model weights are accessible (local or remote).
   - Edit the model path variables in `finalstreamlitui.py` if you need to point at a different location or a remote filesystem.
   - Run the app from the command line:
     ```
     python finalstreamlitui.py
     ```
   - A Streamlit web interface will open (typically at http://localhost:8501) showing the system in action; conversations are streamed in real time.
   - Use this UI to interact with the simulated personas and inspect outputs.

> The pipeline relies on the dictionary, persona definitions and ontology files for simulation and topic validation. Make sure those JSON files are present in the repository root when executing either the notebook or the Streamlit app.
