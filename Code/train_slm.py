# ----------------------------------------------------
# Imports
# ----------------------------------------------------

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from tqdm import tqdm
from component import utils_slm

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
SEED = 42
MAX_LENGTH = 300
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 6
VAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

USE_LORA = True
USE_4BIT = False

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, "Data")

TRAIN_CSV = os.path.join(data_directory, "train.csv")
VAL_CSV = os.path.join(data_directory, "val.csv")

print(f"Data Directory: {data_directory}")

# ----------------------------------------------------
# Setup
# ----------------------------------------------------
utils_slm.set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
print(f"Loading TRAIN split from: {TRAIN_CSV}")
train_df = pd.read_csv(TRAIN_CSV)

print(f"Loading VALIDATION split from: {VAL_CSV}")
val_df = pd.read_csv(VAL_CSV)

print(f"Full Train rows: {len(train_df)}, Full Val rows: {len(val_df)}")

needed_cols = ["story", "words", "story_beginning_prompt"]
train_df = train_df[needed_cols]
val_df = val_df[needed_cols]

print(f"After column filtering - Train rows: {len(train_df)}, Val rows: {len(val_df)}")

# ----------------------------------------------------
# Format Data
# ----------------------------------------------------
print("Formatting text for TRAIN split...")
train_df["text"] = train_df.apply(utils_slm.format_example, axis=1)

print("Formatting text for VALIDATION split...")
val_df["text"] = val_df.apply(utils_slm.format_example, axis=1)

# ----------------------------------------------------
# Create Datasets
# ----------------------------------------------------
train_ds = Dataset.from_pandas(train_df[["text"]])
val_ds = Dataset.from_pandas(val_df[["text"]])

dataset = DatasetDict({"train": train_ds, "validation": val_ds})

print(f"HF Datasets - Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# ----------------------------------------------------
# Load Model & Tokenizer
# ----------------------------------------------------
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading model: {MODEL_NAME}")

if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

model.gradient_checkpointing_enable()

if torch.cuda.is_available():
    print(f"\nGPU Memory Status:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"  Max Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ----------------------------------------------------
# Setup LoRA
# ----------------------------------------------------
if USE_LORA:
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ----------------------------------------------------
# Tokenize Datasets
# ----------------------------------------------------
print("Tokenizing datasets...")
tokenize_fn = utils_slm.tokenize_function(tokenizer, MAX_LENGTH)
tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------------------------------
# Create DataLoaders
# ----------------------------------------------------
tokenized_datasets = tokenized_datasets.with_format("torch")


train_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=2,  # ← Reduced from 4 (safer)
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,  # ← Keep workers alive between epochs
)

val_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=2,
    pin_memory=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# ----------------------------------------------------
# Training Loop
# ----------------------------------------------------
def run_epoch(dataloader, train: bool = True):
    if train:
        model.train()
        phase = "Train"
    else:
        model.eval()
        phase = "Val"

    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        if train:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                )
                loss = outputs.loss

        non_pad = (labels != -100).sum().item()
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS * non_pad
        total_tokens += non_pad

    if train and (step + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(total_tokens, 1)


best_val_loss = float("inf")
best_model_dir = "./story-mistral-best"

print("\nStarting training...", flush=True)
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    print(
        f"Epoch {epoch}/{NUM_EPOCHS} - Train loss/token: {train_loss:.4f}, Val loss/token: {val_loss:.4f}",
        flush=True,
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"New best model at epoch {epoch}, saving to {best_model_dir}...", flush=True)
        os.makedirs(best_model_dir, exist_ok=True)
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

# ----------------------------------------------------
# Save Final Model
# ----------------------------------------------------
print("Saving final model...")
final_dir = "./story-mistral-final"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Final model saved to {final_dir}")

print("\nTraining complete!")