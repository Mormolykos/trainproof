import os
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

# === Paths ===
model_path = "C:/Users/User/Desktop/web_chat_ai"  # Where model.safetensors, config.json, tokenizer.model are
output_dir = "C:/Users/User/Desktop/web_chat_ai/web_chat_finetuned"  # Where fine-tuned model will be saved

# === Load model and tokenizer ===
print("\u2728 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# === Load dataset ===
print("\u2728 Loading dataset...")
dataset = load_dataset("json", data_files="C:/Users/User/Desktop/web_chat_ai/web_chat_multilingual.jsonl", split="train")

# === Tokenize ===
def tokenize_function(example):
    return tokenizer(example["instruction"], truncation=True, padding="max_length", max_length=512)

print("\u2728 Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    evaluation_strategy="no",
    fp16=False,
    report_to=[]  # No wandb
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Train ===
print("\u2728 Starting training...")
trainer.train()

# === Save final model ===
print("\u2728 Saving model...")
trainer.save_model(output_dir)

print("\ud83c\udf1f Training complete! New model saved at:", output_dir)