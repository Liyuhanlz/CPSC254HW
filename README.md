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
pip install torch torchtext scikit-learn pandas numpy python-dotenv
pip install langchain langchain-openai langchain-huggingface
pip install langchain-chroma chromadb pypdf openai
```

## Required Files

Place the following files in the project directory before running:

- CPSC 254_SYL.pdf
- AG News dataset (automatically downloaded when required)
- .env file containing:

```text
OPENAI_API_KEY=YOUR_API_KEY
```

Do not commit API keys to GitHub.
```

## Running the Programs

### Part 1: Text Classification

```bash
python topic_classifier.py
```

### Part 2: LLM Classification Agent

```bash
python classification_agent.py
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

Speech-to-text:

```bash
python stt_rag.py
```

Text-to-speech:

```bash
python tts_rag.py
```

## Notes

- The RAG system uses GPT-OSS and a Chroma vector database built from the course syllabus.
- The Chroma database is saved locally and reused on subsequent runs.
- API keys, generated databases, and temporary files should not be committed to the repository.

## Contribution

Equal contribution by all team members.