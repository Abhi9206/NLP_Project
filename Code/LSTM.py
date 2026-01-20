import os
import ast
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import evaluation_metric as eval_utils  # evaluation utilities
from tqdm import tqdm  # <-- progress bar like baseline

# ----------------------------------------------------
# 0. Config & Paths
# ----------------------------------------------------
SEED = 42
MODEL_NAME = "gpt2"          # tokenizer only
MAX_TOKENS_SRC = 128         # max tokens for encoder input
MAX_TOKENS_TGT = 192         # max tokens for decoder (story)
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0

# NOTE: now we train and evaluate on FULL data (no row limits)
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, "Data")
print(f"Data Directory: {data_directory}")

TRAIN_CSV = os.path.join(data_directory, "train.csv")
VAL_CSV = os.path.join(data_directory, "val.csv")

SAVE_DIR = "./lstm_attention_story"  # where we save this model


# ----------------------------------------------------
# 1. Seeding & device
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
# 2. Load train & val data (FULL data, no subsampling)
# ----------------------------------------------------
def load_splits(train_csv: str, val_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train file not found at: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Val file not found at: {val_csv}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"Full train rows: {len(train_df)}, full val rows: {len(val_df)}")

    needed_cols = ["story", "words", "story_beginning_prompt"]
    train_df = train_df[needed_cols].reset_index(drop=True)
    val_df = val_df[needed_cols].reset_index(drop=True)

    print(f"Using train rows: {len(train_df)}, val rows: {len(val_df)}")
    return train_df, val_df


train_df, val_df = load_splits(TRAIN_CSV, VAL_CSV)


# ----------------------------------------------------
# 3. Tokenizer (GPT-2 tokenizer, used for IDs only)
# ----------------------------------------------------
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# create a REAL pad token separate from EOS for better training
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

PAD_ID = tokenizer.pad_token_id
BOS_ID = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
EOS_ID = tokenizer.eos_token_id
VOCAB_SIZE = len(tokenizer)  # after possibly adding pad token


# ----------------------------------------------------
# 4. Build text fields: encoder input prompt, decoder target story
# ----------------------------------------------------
def build_prompt(row) -> str:
    """Use same style as your previous prompts."""
    if isinstance(row["words"], str) and row["words"].startswith("["):
        words_list = ast.literal_eval(row["words"])
    else:
        words_list = row["words"]

    if not isinstance(words_list, list):
        words_list = []

    keywords = ", ".join(str(w) for w in words_list)
    prompt = str(row["story_beginning_prompt"]).strip()

    text = (
        "You are a children’s story writer.\n"
        f"Use these words: {keywords}\n"
        f"Story prompt: {prompt}\n"
        "Write a complete story:\n"
    )
    return text


train_df["src_text"] = train_df.apply(build_prompt, axis=1)
train_df["tgt_text"] = train_df["story"].astype(str).apply(lambda s: s.strip())

val_df["src_text"] = val_df.apply(build_prompt, axis=1)
val_df["tgt_text"] = val_df["story"].astype(str).apply(lambda s: s.strip())


# ----------------------------------------------------
# 5. PyTorch Dataset & DataLoader
# ----------------------------------------------------
class StorySeq2SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.src_texts = df["src_text"].tolist()
        self.tgt_texts = df["tgt_text"].tolist()

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return {
            "src_text": self.src_texts[idx],
            "tgt_text": self.tgt_texts[idx],
        }


def collate_fn(batch: List[dict]):
    src_texts = [b["src_text"] for b in batch]
    tgt_texts = [b["tgt_text"] for b in batch]

    # Encode source (encoder input)
    src_enc = tokenizer(
        src_texts,
        truncation=True,
        max_length=MAX_TOKENS_SRC,
        padding=True,
        return_tensors="pt",
    )

    # Encode target (decoder input/output), add BOS/EOS
    tgt_ids_list = []
    for t in tgt_texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        ids = ids[: (MAX_TOKENS_TGT - 2)]  # room for BOS/EOS
        ids = [BOS_ID] + ids + [EOS_ID]
        tgt_ids_list.append(torch.tensor(ids, dtype=torch.long))

    tgt_padded = nn.utils.rnn.pad_sequence(
        tgt_ids_list, batch_first=True, padding_value=PAD_ID
    )

    src_input_ids = src_enc["input_ids"]
    src_attention_mask = src_enc["attention_mask"]

    # Decoder input: everything except last token
    dec_input_ids = tgt_padded[:, :-1]
    # Decoder target: everything except first token
    dec_target_ids = tgt_padded[:, 1:]

    return (
        src_input_ids,
        src_attention_mask,
        dec_input_ids,
        dec_target_ids,
    )


train_dataset = StorySeq2SeqDataset(train_df)
val_dataset = StorySeq2SeqDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# ----------------------------------------------------
# 6. LSTM + Attention Model
# ----------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(self, src_ids, src_mask):
        embedded = self.embedding(src_ids)  # (B, T, E)
        outputs, (h, c) = self.lstm(embedded)  # outputs: (B, T, H)
        return outputs, (h, c)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: (B, H)
        encoder_outputs: (B, T, H)
        mask: (B, T)  1 for valid, 0 for pad (optional)
        """
        dec_hidden_exp = decoder_hidden.unsqueeze(1).expand_as(encoder_outputs)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(dec_hidden_exp))).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(score, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B, 1, H)
        context = context.squeeze(1)  # (B, H)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,  # concat [embed, context]
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = BahdanauAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, dec_input_ids, initial_hidden, initial_cell, encoder_outputs, src_mask):
        embedded = self.embedding(dec_input_ids)  # (B, T_dec, E)
        B, T_dec, _ = embedded.size()
        h, c = initial_hidden, initial_cell

        logits = []
        for t in range(T_dec):
            emb_t = embedded[:, t, :]  # (B, E)
            context, _ = self.attention(h[-1], encoder_outputs, src_mask)  # (B, H)
            lstm_input = torch.cat([emb_t, context], dim=-1).unsqueeze(1)  # (B, 1, E+H)
            output, (h, c) = self.lstm(lstm_input, (h, c))  # (B, 1, H)
            step_logits = self.fc_out(output.squeeze(1))  # (B, vocab)
            logits.append(step_logits.unsqueeze(1))

        logits = torch.cat(logits, dim=1)  # (B, T_dec, vocab)
        return logits


class Seq2SeqAttnModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, src_ids, src_mask, dec_input_ids, dec_target_ids=None):
        enc_outputs, (h, c) = self.encoder(src_ids, src_mask)
        logits = self.decoder(dec_input_ids, h, c, enc_outputs, src_mask)
        return logits


model = Seq2SeqAttnModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

print(model)


# ----------------------------------------------------
# 7. Training Loop (with tqdm + best model saving)
# ----------------------------------------------------
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def run_epoch(dataloader, train: bool = True):
    if train:
        model.train()
        phase = "Train"
    else:
        model.eval()
        phase = "Val"

    epoch_loss = 0.0
    total_tokens = 0

    # tqdm progress bar, like baseline
    for batch in tqdm(dataloader, desc=phase):
        src_ids, src_mask, dec_input_ids, dec_target_ids = batch
        src_ids = src_ids.to(device)
        src_mask = src_mask.to(device)
        dec_input_ids = dec_input_ids.to(device)
        dec_target_ids = dec_target_ids.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(src_ids, src_mask, dec_input_ids)  # (B, T_dec, vocab)
        B, T, V = logits.size()

        loss = criterion(
            logits.view(B * T, V),
            dec_target_ids.reshape(B * T),
        )

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        non_pad = (dec_target_ids != PAD_ID).sum().item()
        epoch_loss += loss.item() * non_pad
        total_tokens += non_pad

    return epoch_loss / max(total_tokens, 1)


# best model tracking (like GPT-2 baseline)
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_SAVE_PATH = os.path.join(SAVE_DIR, "lstm_attention_best.pt")
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, "lstm_attention_final.pt")

best_val_loss = float("inf")

print("\nStarting training...", flush=True)
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    print(
        f"Epoch {epoch}/{NUM_EPOCHS} - Train loss/token: {train_loss:.4f}, "
        f"Val loss/token: {val_loss:.4f}",
        flush=True,
    )

    # save best model by validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"New best model at epoch {epoch}, saving checkpoint to {BEST_SAVE_PATH} ...", flush=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": VOCAB_SIZE,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "tokenizer_name": MODEL_NAME,
            },
            BEST_SAVE_PATH,
        )

print("\nSaving final (last-epoch) model...", flush=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "tokenizer_name": MODEL_NAME,
    },
    FINAL_SAVE_PATH,
)
print(f"Final model saved to {FINAL_SAVE_PATH}", flush=True)

print(f"\nReloading best model from {BEST_SAVE_PATH} for generation & evaluation...", flush=True)
best_ckpt = torch.load(BEST_SAVE_PATH, map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
model.to(device)


# ----------------------------------------------------
# 9. Build generation prompt (same style as baseline)
# ----------------------------------------------------
def build_generation_prompt(row):
    """
    Build the prompt WITHOUT appending the gold story.
    This will be fed to the LSTM model for generation on validation set.
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


# ----------------------------------------------------
# 10. LSTM generation (greedy decoding with attention)
# ----------------------------------------------------
def generate_story_lstm(prompt, max_new_tokens=128):
    """
    Generate a story continuation from the given prompt using the LSTM+Attention model.
    Returns only the generated continuation (no prompt).
    """
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS_SRC,
        )
        src_ids = enc["input_ids"].to(device)        # (1, T_src)
        src_mask = enc["attention_mask"].to(device)  # (1, T_src)

        encoder_outputs, (h0, c0) = model.encoder(src_ids, src_mask)

        dec_input_ids = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
        generated_ids = []

        for _ in range(max_new_tokens):
            logits = model.decoder(dec_input_ids, h0, c0, encoder_outputs, src_mask)  # (1, T_dec, vocab)
            next_logits = logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1)

            token_id = next_id.item()
            if token_id == EOS_ID:
                break

            generated_ids.append(token_id)
            dec_input_ids = torch.cat([dec_input_ids, next_id.unsqueeze(0)], dim=1)

        if not generated_ids:
            return ""

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()


# ----------------------------------------------------
# 11. Helper: clean/parse 'words' column for keyword metrics
# ----------------------------------------------------
def parse_words_column(x):
    """
    Convert the 'words' field into a clean list of keyword strings.
    Handles:
      - string representations like "['cat', 'dog']"
      - simple comma-separated strings like "cat, dog"
      - list objects
      - NaN / None / empty values
    Returns: List[str]
    """
    # Case 1: already a list
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
# 12. Generate on FULL VALIDATION set & evaluate (with keyword metrics)
# ----------------------------------------------------
print("Generating stories for FULL VALIDATION set with LSTM+Attention...")
val_eval_df = val_df.copy().reset_index(drop=True)
print(f"Evaluating on all validation rows: {len(val_eval_df)}")

generated_stories = []
for i, row in val_eval_df.iterrows():
    print(f"Generating story {i}/{len(val_eval_df)}...")
    prompt = build_generation_prompt(row)
    gen_story = generate_story_lstm(prompt)
    generated_stories.append(gen_story)

# Clean 'words' so keyword metrics work properly
val_eval_df["words_clean"] = val_eval_df["words"].apply(parse_words_column)

eval_df = pd.DataFrame(
    {
        "id": range(len(val_eval_df)),
        "story": val_eval_df["story"].astype(str).tolist(),
        "generated_story": generated_stories,
        "words": val_eval_df["words_clean"].tolist(),  # cleaned keywords for keyword metrics
    }
)

print("Running evaluation metrics on FULL VALIDATION set (LSTM+Attention)...")
eval_df_with_metrics, summary = eval_utils.evaluate_text_generation(
    eval_df,
    ref_col="story",
    hyp_col="generated_story",
)

print("\nValidation Evaluation Summary (LSTM+Attention, full val):")
for k, v in summary.items():
    try:
        print(f"{k}: {v:.4f}")
    except TypeError:
        print(f"{k}: {v}")

eval_df_with_metrics.to_csv("evaluation_results_val_lstm_attention_full.csv", index=False)
print("\nSaved per-example validation metrics to evaluation_results_val_lstm_attention_full.csv")
