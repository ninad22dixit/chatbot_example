# Domain-Specific Chatbot with Retrieval-Augmented Generation (RAG)

This project is a lightweight, domain-adapted chatbot you can run **locally on CPU**.  

It uses:
- **LoRA** for parameter-efficient fine-tuning on the domain data (e.g. Q&A pairs)
- **SentenceTransformers + FAISS** for retrieval-augmented generation (RAG)
- A **FastAPI** server for an easy-to-use local HTTP API with built-in Swagger UI

---

## Features

* Fine-tune open-weight LLM: TinyLlama-1.1B-Chat-v1.0  
* Retrieval-augmented generation using local document store (FAISS)   
* CPU-friendly default setup  
* REST API  

---

## Project Structure
├── lora_training.py # Fine-tunes the base model with LoRA

├── retriever.py # Builds FAISS index from the dataset

├── chatbot_rag.py # Runs chatbot directly in the terminal

├── api_server.py # FastAPI server for interactive chatbot

└── requirements.txt # Dependencies

## Requirements

- Python 3.8+
- CPU (default)
- Or GPU (optional, for faster training/inference)

---

## Installation

Clone this repo:

    git clone https://github.com/ninad22dixit/chatbot_example.git
    cd chatbot_example

## Install requirements

    pip install -r requirements.txt

## Hugging Face Access

Make sure you have a Hugging Face account and a token.

## Quickstart: CPU-only Local Deployment

This project is set up by default for CPU.

## Fine-tune with LoRA

    python lora_training.py

Loads ~2000 samples from squad_v2 (or modify to your domain). Saves LoRA-adapted model to ./lora-finetuned. This step will take a long time when using only CPU.

## Build the Retriever

    python retriever.py

Builds FAISS index with SentenceTransformer embeddings

Saves index and corpus locally

## Start the Chatbot API

- Run in terminal

        uvicorn api_server:app --host 127.0.0.1 --port 8000

- Open your browser at:

        http://localhost:8000/docs

- Explore endpoints

- Try out /query with example questions

- See responses including retrieved context


## License
MIT. Free to use, adapt, and share.

## Acknowledgements
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

- [PEFT for LoRA](https://huggingface.co/docs/peft/index)

- [SentenceTransformers](https://www.sbert.net/)

- [FAISS](https://github.com/facebookresearch/faiss)

- [FastAPI](https://fastapi.tiangolo.com/)
