import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from tqdm import tqdm
import random

# Path Set-Up
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
code_directory = os.path.join(parent_directory, "Code")
data_directory = os.path.join(parent_directory, "Data")

TRAIN_CSV = os.path.join(data_directory, "train.csv")
VAL_CSV = os.path.join(data_directory, "val.csv")
MODEL_SAVE_DIR = os.path.join(code_directory, "story-flan-t5-lora")

BASE_MODEL = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Training data: {TRAIN_CSV}")
print(f"Validation data: {VAL_CSV}")
print(f"Save dir: {MODEL_SAVE_DIR}")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#Create Dataset
class StoryDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_input_length=128, max_output_length=400):
        self.df = pd.read_csv(csv_path).dropna(
            subset=["story_beginning_prompt", "story"]
        ).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract keywords
        words = []
        if pd.notna(row.get("words")):
            try:
                if isinstance(row["words"], str) and row["words"].startswith("["):
                    words = ast.literal_eval(row["words"])
            except:
                pass

        keywords = ", ".join(words) if words else "happy, friend, sun"
        prompt = (
            f"You are a kind children's story writer.\n"
            f"Use these words: {keywords}\n"
            f"Start with: {row['story_beginning_prompt']}\n"
            f"Write a complete short story:"
        )

        # Tokenize inputs and outputs
        input_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        ).input_ids.squeeze()

        labels = tokenizer(
            row['story'],
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Replace pad token with -100 for ignoring in loss
        labels[labels == tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "labels": labels}

#Dataloaders

train_dataset = StoryDataset(TRAIN_CSV, tokenizer)
val_dataset = StoryDataset(VAL_CSV, tokenizer)

def collate_fn(batch):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True,
            padding_value=-100
        )
    }

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

#Load Model with LoRa
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.train()
model.to(DEVICE)


#Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
num_training_steps = len(train_loader) * 3  # epochs = 3
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

#Training Loop

EPOCHS = 3
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    loop = tqdm(train_loader, leave=False)
    total_loss = 0

    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss):
            print("NaN loss detected. Skipping step.")
            continue

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        loop.set_description(f"loss={loss.item():.4f}")

    print(f"Average training loss: {total_loss/len(train_loader):.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"epoch_{epoch+1}")
    model.save_pretrained(checkpoint_path)

print("\nTraining completed. Final model saved.")
