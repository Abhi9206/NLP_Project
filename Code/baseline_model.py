# baseline.py (or train_val.py)

import os
import random
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
import evaluation_metric as eval_utils
from tqdm import tqdm

# ----------------------------------------------------
# 0. Config & Paths
# ----------------------------------------------------
SEED = 42
MAX_LENGTH = 192        # you can bump to 256 later if GPU allows
MODEL_NAME = "gpt2"
NUM_EPOCHS = 5          # train for 5 epochs
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2

# Project structure: /home/ubuntu/Project/{Code, Data}
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, "Data")
print(f"Data Directory: {data_directory}")

TRAIN_CSV = os.path.join(data_directory, "train.csv")
VAL_CSV   = os.path.join(data_directory, "val.csv")


# ----------------------------------------------------
# 1. Seeding (make results as deterministic as possible)
# ----------------------------------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ----------------------------------------------------
# 2. Load train & val splits (USE ALL ROWS)
# ----------------------------------------------------
print(f"Loading TRAIN split from: {TRAIN_CSV}")
train_df = pd.read_csv(TRAIN_CSV)

print(f"Loading VALIDATION split from: {VAL_CSV}")
val_df = pd.read_csv(VAL_CSV)

print(f"Full Train rows: {len(train_df)}, Full Val rows: {len(val_df)}")

# Keep only the columns we actually need (saves memory)
needed_cols = ["story", "words", "story_beginning_prompt"]
train_df = train_df[needed_cols]
val_df = val_df[needed_cols]

print(f"After column filtering - Train rows: {len(train_df)}, Val rows: {len(val_df)}")


# ----------------------------------------------------
# 3. Build training text (prompt + gold story)
# ----------------------------------------------------
def format_example(row):
    """Create full training text (prompt + gold story)."""
    if isinstance(row["words"], str) and row["words"].startswith("["):
        words_list = ast.literal_eval(row["words"])
    else:
        words_list = row["words"]

    # Fall back to empty list if something went wrong
    if not isinstance(words_list, list):
        words_list = []

    keywords_str = ", ".join(str(w) for w in words_list)
    prompt_str = str(row["story_beginning_prompt"]).strip()
    story_str = str(row["story"]).strip()

    input_text = (
        "You are a children’s story writer.\n"
        f"Use these words: {keywords_str}\n"
        f"Story prompt: {prompt_str}\n"
        "Write a complete story:\n"
    )
    return input_text + story_str


print("Formatting text for TRAIN split...")
train_df["text"] = train_df.apply(format_example, axis=1)

print("Formatting text for VALIDATION split...")
val_df["text"] = val_df.apply(format_example, axis=1)


# ----------------------------------------------------
# 4. Wrap into Hugging Face Datasets
# ----------------------------------------------------
train_ds = Dataset.from_pandas(train_df[["text"]])
val_ds = Dataset.from_pandas(val_df[["text"]])

dataset = DatasetDict(
    {
        "train": train_ds,
        "validation": val_ds,
    }
)

print(f"HF Datasets - Train size: {len(train_ds)}, Val size: {len(val_ds)}")


# ----------------------------------------------------
# 5. Load tokenizer & model
# ----------------------------------------------------
print(f"Loading tokenizer & model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        # padding is handled by data_collator
    )


print("Tokenizing datasets...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ----------------------------------------------------
# 6. DataLoaders for manual training loop
# ----------------------------------------------------
train_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)

val_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

# smaller LR for more stable training
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


# ----------------------------------------------------
# 7. Manual training loop with per-epoch loss + best model saving
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

    # tqdm progress bar
    for batch in tqdm(dataloader, desc=phase):
        # batch has: input_ids, attention_mask, labels (from DataCollatorForLanguageModeling)
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]  # shape (B, T), ignore_index = -100

        if train:
            optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss  # average per non-ignored token

        if train:
            loss.backward()
            optimizer.step()

        non_pad = (labels != -100).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)

best_val_loss = float("inf")
best_model_dir = "./story-gpt2-best"

print("\nStarting training...", flush=True)
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Train loss/token: {train_loss:.4f}, Val loss/token: {val_loss:.4f}",flush=True,)

    # save best model by validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"New best model at epoch {epoch}, saving checkpoint to {best_model_dir} ...", flush=True)
        os.makedirs(best_model_dir, exist_ok=True)
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)


# ----------------------------------------------------
# 8. Save final model & reload best model for generation
# ----------------------------------------------------
print("Saving final (last-epoch) model...")
final_dir = "./story-gpt2-final"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Final model saved to {final_dir}")

print(f"Reloading best model from {best_model_dir} for generation & evaluation...")
model = AutoModelForCausalLM.from_pretrained(best_model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ----------------------------------------------------
# 9. Build generation prompt (NO gold story)
# ----------------------------------------------------
def build_generation_prompt(row):
    """
    Build the prompt WITHOUT appending the gold story.
    This will be fed to the model for generation on validation set.
    """
    if isinstance(row["words"], str) and row["words"].startswith("["):
        words_list = ast.literal_eval(row["words"])
    else:
        words_list = row["words"]

    if not isinstance(words_list, list):
        words_list = []

    keywords_str = ", ".join(str(w) for w in words_list)
    prompt_str = str(row["story_beginning_prompt"]).strip()

    prompt = (
        "You are a children’s story writer.\n"
        f"Use these words: {keywords_str}\n"
        f"Story prompt: {prompt_str}\n"
        "Write a complete story:\n"
    )
    return prompt


def generate_story(prompt, max_new_tokens=256, temperature=0.6, top_p=0.95):
    """
    Generate a story continuation from the given prompt.
    Returns only the generated continuation (prompt stripped off).
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,   # reduce repetition a bit
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if full_text.startswith(prompt_text):
            generated = full_text[len(prompt_text):].strip()
        else:
            generated = full_text.strip()

    return generated


# ----------------------------------------------------
# Helper: clean/parse 'words' column for keyword metrics
# ----------------------------------------------------
def parse_words_column(x):
    if isinstance(x, list):
        return [str(w).strip().lower() for w in x if str(w).strip()]

    # Case 2: string
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan":
            return []
        try:
            # Looks like a Python list string
            if s.startswith("[") and s.endswith("]"):
                lst = ast.literal_eval(s)
            else:
                # Fallback: comma-separated keywords
                lst = s.split(",")
        except Exception:
            lst = []
        return [str(w).strip().lower() for w in lst if str(w).strip()]

    # Case 3: anything else (NaN, None, etc.)
    return []


# ----------------------------------------------------
# 10. Generate on VALIDATION set (ALL ROWS) & evaluate
# ----------------------------------------------------
print("Generating stories for VALIDATION set (ALL rows)...")

# Use ALL rows of the validation dataframe
val_df = val_df.reset_index(drop=True)

generated_stories = []
for i, row in val_df.iterrows():
    if i % 50 == 0:
        print(f"Generating story {i}/{len(val_df)}...")
    prompt = build_generation_prompt(row)
    gen_story = generate_story(prompt)
    generated_stories.append(gen_story)

# Clean 'words' so keyword metrics work properly
val_df["words_clean"] = val_df["words"].apply(parse_words_column)

eval_df = pd.DataFrame(
    {
        "id": range(len(val_df)),
        "story": val_df["story"].astype(str).tolist(),
        "generated_story": generated_stories,
        "words": val_df["words_clean"].tolist(),  # cleaned keywords (per-row)
    }
)

print("Running evaluation metrics on FULL VALIDATION set...")
eval_df_with_metrics, summary = eval_utils.evaluate_text_generation(
    eval_df,
    ref_col="story",
    hyp_col="generated_story",
)

print("\nValidation Evaluation Summary (FULL validation set):")
for k, v in summary.items():
    try:
        print(f"{k}: {v:.4f}")
    except TypeError:
        print(f"{k}: {v}")

eval_df_with_metrics.to_csv("evaluation_results_val_full.csv", index=False)
print("\nSaved per-example validation metrics to evaluation_results_val_full.csv")
