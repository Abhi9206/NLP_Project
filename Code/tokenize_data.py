# --------------------------------------------------------------------------------------------------------------------
# Import Relevant Packages
# --------------------------------------------------------------------------------------------------------------------
# %%

import pandas as pd
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import json

from component.utils import tokenize_split



#--------------------------------------------------------------------------------------------------------------------
# Set Working Directory
#--------------------------------------------------------------------------------------------------------------------
#%%

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data/')
token_directory = os.path.join(parent_directory, 'Tokenization/')
print(f"Data Directory: {data_directory}")


#--------------------------------------------------------------------------------------------------------------------
# Load Tokenizer
#--------------------------------------------------------------------------------------------------------------------
#%%


print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


#--------------------------------------------------------------------------------------------------------------------
# Load Datasets
#--------------------------------------------------------------------------------------------------------------------
#%%

print("\nLoading datasets...")
train_df = pd.read_csv(data_directory + 'train.csv')
val_df = pd.read_csv(data_directory + 'val.csv')
test_df = pd.read_csv(data_directory + 'test.csv')

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


#--------------------------------------------------------------------------------------------------------------------
# Tokenize Data
#--------------------------------------------------------------------------------------------------------------------
#%%

train_tokenized = tokenize_split(train_df, token_directory + 'train')
val_tokenized = tokenize_split(val_df, token_directory + 'val')
test_tokenized = tokenize_split(test_df, token_directory + 'test')

#--------------------------------------------------------------------------------------------------------------------
# Calculate Statistics
#--------------------------------------------------------------------------------------------------------------------
#%%

print("\nCalculating statistics...")

# Train stats
train_input_lens = [len(x['input_ids']) for x in train_tokenized]
train_target_lens = [len(x['target_ids']) for x in train_tokenized]
print(
    f"Train: Input mean={sum(train_input_lens) / len(train_input_lens):.1f}, max={max(train_input_lens)} | Target mean={sum(train_target_lens) / len(train_target_lens):.1f}, max={max(train_target_lens)}")

# Val stats
val_input_lens = [len(x['input_ids']) for x in val_tokenized]
val_target_lens = [len(x['target_ids']) for x in val_tokenized]
print(
    f"Val: Input mean={sum(val_input_lens) / len(val_input_lens):.1f}, max={max(val_input_lens)} | Target mean={sum(val_target_lens) / len(val_target_lens):.1f}, max={max(val_target_lens)}")

# Test stats
test_input_lens = [len(x['input_ids']) for x in test_tokenized]
test_target_lens = [len(x['target_ids']) for x in test_tokenized]
print(
    f"Test: Input mean={sum(test_input_lens) / len(test_input_lens):.1f}, max={max(test_input_lens)} | Target mean={sum(test_target_lens) / len(test_target_lens):.1f}, max={max(test_target_lens)}")


#--------------------------------------------------------------------------------------------------------------------
# Save Files
#--------------------------------------------------------------------------------------------------------------------
#%%

print("\nSaving files...")
torch.save(train_tokenized, token_directory + 'train_tokenized.pt')
torch.save(val_tokenized, token_directory + 'val_tokenized.pt')
torch.save(test_tokenized, token_directory + 'test_tokenized.pt')
tokenizer.save_pretrained(token_directory + 'tokenizer')

# Save metadata
metadata = {
    'vocab_size': len(tokenizer),
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'train_size': len(train_tokenized),
    'val_size': len(val_tokenized),
    'test_size': len(test_tokenized)
}
with open(token_directory + 'tokenization_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nDone!")
print(f"Files saved!")
