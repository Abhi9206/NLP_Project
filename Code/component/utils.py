#--------------------------------------------------------------------------------------------------------------------
# Import Relevant Packages
#--------------------------------------------------------------------------------------------------------------------
#%%

import sys
import os
import glob
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
import torch
from transformers import GPT2Tokenizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")



#--------------------------------------------------------------------------------------------------------------------
# Function to Read and Merge JSON Files
#--------------------------------------------------------------------------------------------------------------------
#%%

def read_and_merge_json_files(data_directory):
    """Read and merge JSON files into a single DataFrame."""

    directory = Path(data_directory)
    json_files = list(directory.glob("*.json"))
    json_files.sort()

    df_list = []

    pbar = tqdm(json_files, desc="Processing JSON files")
    for json_file in pbar:
        try:
            df = pd.read_json(json_file)
            df_list.append(df)
            pbar.set_description(f"✓ {json_file.name[:30]}")
            pbar.set_postfix({"rows": len(df)})

        except Exception as e:
            pbar.set_description(f"✗ Error: {json_file.name[:30]}")
            print(f"\nError: Could not read {json_file.name}: {e}")
            continue

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.astype(str).drop_duplicates()
    final_df = final_df.reset_index(drop=True)
    total_observations = len(final_df)

    return total_observations, final_df





#--------------------------------------------------------------------------------------------------------------------
# Function to Create Partial Sentence
#--------------------------------------------------------------------------------------------------------------------
#%%

def create_partial_sentence(story, num_words=7):
    """Extract first N words from story as partial sentence"""
    if pd.isna(story):
        return ""

    words = story.split()
    partial = ' '.join(words[:num_words])
    return partial



#--------------------------------------------------------------------------------------------------------------------
# Function to Count Words
#--------------------------------------------------------------------------------------------------------------------
#%%

def count_words(text):
    """Count number of words in text"""
    if pd.isna(text):
        return 0
    return len(text.split())

#--------------------------------------------------------------------------------------------------------------------
# Function to Count Characters
#--------------------------------------------------------------------------------------------------------------------
#%%

def normalize_whitespace(text):
    """Normalize all whitespace in text"""
    if pd.isna(text):
        return ""

    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


#--------------------------------------------------------------------------------------------------------------------
# Function to Format Input
#--------------------------------------------------------------------------------------------------------------------

def format_input(partial, keywords):
    """
    Format input text consistently across all approaches.
    Used during training and inference.

    Args:
        partial: Partial sentence string
        keywords: List of keyword strings

    Returns:
        input_text: Formatted input string
    """
    keywords_str = ', '.join(keywords)
    input_text = f"Keywords: {keywords_str} Story: {partial}"
    return input_text


#--------------------------------------------------------------------------------------------------------------------
# Function to Load Tokenized Data
#--------------------------------------------------------------------------------------------------------------------

def load_tokenized_data(split='train'):
    """
    Load pre-tokenized dataset.
    Used by all 3 approaches during training.

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        tokenized_data: List of tokenized samples
    """
    filepath = f'{split}_tokenized.pt'
    tokenized_data = torch.load(filepath, weights_only=False)
    return tokenized_data


#--------------------------------------------------------------------------------------------------------------------
# Function to Tokenize Split
#--------------------------------------------------------------------------------------------------------------------

def tokenize_split(df, split_name):
    """Tokenize a single split"""

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {split_name.split('/')[-1]}"):
        keywords = eval(row['words']) if isinstance(row['words'], str) else row['words']
        input_text = format_input(row['story_beginning_prompt'], keywords)

        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        target_ids = tokenizer.encode(row['story'], add_special_tokens=True)

        tokenized.append({
            'input_ids': input_ids,
            'target_ids': target_ids,
            'input_text': input_text,
            'target_text': row['story'],
            'keywords': keywords
        })
    return tokenized