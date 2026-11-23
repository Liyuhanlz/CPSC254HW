# CPSC 254 Assignment 4
**Team Members:** Liyuhan Zhou, Jessica Mijares, Nikita Subedi

## Project Overview
This project contains four parts:

1. Text preprocessing, vectorization, and classification using the AG News dataset.
2. LLM-based news classification using the GPT-OSS model.
3. Retrieval-Augmented Generation (RAG) using the course syllabus PDF and Chroma vector database.
4. Speech-to-text (STT) and text-to-speech (TTS) extensions for the RAG system.

## Environment Assumptions

- Python 3.10+
- Internet access for GPT-OSS API requests
- Access to the Nautilus Hypercluster endpoint
- Windows environment was used for development and testing

## Dependencies

Install the required packages before running the programs:

```bash
pip install torch torchtext torchdata scikit-learn pandas numpy joblib matplotlib python-dotenv
pip install langchain langchain-openai langchain-community langchain-text-splitters langchain-huggingface sentence-transformers
pip install langchain-chroma chromadb pypdf openai
pip install openai-whisper sounddevice soundfile scipy pyttsx3
```

`torchtext` only works with older PyTorch versions (e.g. torch 2.3 with torchtext 0.18).

## Required Files

Place the following files in the project directory before running:

- CPSC 254_SYL.pdf
- AG News dataset (automatically downloaded when required)
- .env file containing:

```text
OPENAI_API_KEY=YOUR_API_KEY
```

Do not commit API keys to GitHub.

## Running the Programs

Run the programs in order: Part 1 must be run before Part 2 (it creates `best_topic_classifier.joblib` and `sample_idx_200.npy`), and Part 3 must be set up before Part 4.

### Part 1: Text Classification

```bash
python topic_classifier.py
```

### Part 2: LLM Classification Agent

Classify one article:

```bash
python classification_agent.py --text "Oil prices hit record as Gulf storm shuts refineries."
```

Compare the agent with the Part 1 classifier on the 200 shared test examples:

```bash
python classification_agent.py --evaluate
```

### Part 3: RAG Application

Run the RAG system:

```bash
python rag.py
```

Run the evaluation script:

```bash
python rag_eval.py
```

The evaluation script generates:

```text
rag_evaluation_results.csv
```

### Part 4: Speech Recognition and Voice Synthesis

Speech-to-text (create a spoken question audio file, then transcribe it and answer it with the RAG system):

```bash
python stt_rag.py --make-sample "What are the instructor's office hours?" --out audio/question1.wav
python stt_rag.py --audio audio/question1.wav
```

Or record the question from a microphone:

```bash
python stt_rag.py --mic
```

Text-to-speech (answer a question with the RAG system and speak the answer aloud):

```bash
python tts_rag.py "How is the final grade weighted between participation, assignments, and exams?"
```

## Notes

- The RAG system uses GPT-OSS and a Chroma vector database built from the course syllabus.
- The Chroma database is saved locally and reused on subsequent runs.
- API keys, generated databases, and temporary files should not be committed to the repository.

## Contribution

Equal contribution by all team members.