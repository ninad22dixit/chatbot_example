# Import necessary libraries
import torch
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from fastapi.responses import FileResponse


# Load components
print("Loading retriever...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.bin")
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f]

print("Loading fine-tuned model...")
model_name_or_path = "./lora-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# FastAPI setup
app = FastAPI(
    title="Domain-Specific Chatbot with RAG",
    description="""
**Welcome to the Domain Chatbot API!**  
This API provides question-answering over your custom domain using retrieval-augmented generation (RAG).

- Retrieve domain-specific context using SentenceTransformers + FAISS.
- Generate answers using a LoRA-fine-tuned LLM (e.g. Mistral-7B).

Test it live below!
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        example="How do I bake sourdough bread?"
    )

class QueryResponse(BaseModel):
    answer: str = Field(
        ...,
        example="You need flour, water, salt, and starter..."
    )
    retrieved_context: str = Field(
        ...,
        example="You need flour, water, salt, and starter..."
    )


# Helper
def retrieve_context(query, k=1):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [corpus[i] for i in I[0]]

def generate_answer(user_query):
    context = retrieve_context(user_query)[0]
    prompt = f"Answer the question using this context:\n{context}\nQuestion: {user_query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), context


# Routes
@app.get("/favicon.ico")
def favicon():
    return {}

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question to the chatbot",
    description="""
Submit a question as JSON.

The API will:
1. Retrieve relevant context using a local FAISS index.
2. Generate an answer with the fine-tuned LLM.

Returns both the generated answer and the retrieved context.
"""
)
def query_bot(request: QueryRequest):
    answer, context = generate_answer(request.question)
    return QueryResponse(answer=answer, retrieved_context=context)

@app.get("/")
def root():
    return {"message": "Domain-Specific Chatbot with RAG is running!"}
