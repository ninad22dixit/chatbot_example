# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
login()


# Import model and create tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model = prepare_model_for_kbit_training(model)

# Set up LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


# Load External Free Data
dataset = load_dataset("squad_v2", split="train[:2000]")  # small sample
def format_qa(example):
    return {"text": f"Question: {example['question']}\nAnswer: {example['answers']['text'][0] if example['answers']['text'] else 'No answer'}"}
dataset = dataset.map(format_qa)

def tokenize(batch):
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encodings["labels"] = encodings["input_ids"]
    return encodings

dataset = dataset.map(tokenize, batched=True)

# Training step
training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# Save model
model.save_pretrained("./lora-finetuned")
tokenizer.save_pretrained("./lora-finetuned")
print("✅ LoRA fine-tuned model saved to ./lora-finetuned")