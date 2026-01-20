#--------------------------------------------------------------------------------------------------------------------
# Import Relevant Packages
#--------------------------------------------------------------------------------------------------------------------
#%%

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import ast

from component import utils



#--------------------------------------------------------------------------------------------------------------------
# Set Working Directory
#--------------------------------------------------------------------------------------------------------------------
#%%

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')
print(f"Data Directory: {data_directory}")


#--------------------------------------------------------------------------------------------------------------------
# Load Data
#--------------------------------------------------------------------------------------------------------------------
#%%

read_data = utils.read_and_merge_json_files(data_directory)
total_observations, final_df = read_data

final_df = final_df[0:100000]




#--------------------------------------------------------------------------------------------------------------------
# Clean Data
#--------------------------------------------------------------------------------------------------------------------
#%%

final_df['words'] = final_df['instruction'].apply(
    lambda x: ast.literal_eval(x).get('words', []) if pd.notna(x) else []
)

final_df['story_beginning_prompt'] = final_df['story'].apply(
    lambda x: utils.create_partial_sentence(x, 10)
)

final_df['word_count'] = final_df['story'].apply(
    lambda x: utils.count_words(x)
)

final_df['story'] = final_df['story'].apply(
    lambda x: utils.normalize_whitespace(x)
)

final_df = final_df.drop(columns=['summary', 'source', 'instruction'])

null_values = final_df.isnull().sum().to_frame(name='Null Count')
print(f"Null Values:\n{tabulate(null_values, headers='keys', tablefmt='pretty')}")


# get_data_fields = json.load(open(json_files[0]))


#--------------------------------------------------------------------------------------------------------------------
# Preview statistics
#--------------------------------------------------------------------------------------------------------------------
#%%

print("*"*50)
print(f"Dataset Overview(after Filtering)")
print("*"*50)

print(f"Total Observations: {total_observations}")
print(f"\nDataframe Head: \n{tabulate(final_df.head(2), headers='keys', tablefmt='pretty')}")
print(f"\nThe data fields are: {final_df.columns.tolist()}")

print("\nWord Count Statistics:")
print(f"< 20 words: {(final_df['word_count'] < 20).sum()}")
print(f"20-50 words: {((final_df['word_count'] >= 20) & (final_df['word_count'] < 50)).sum()}")
print(f"50-100 words: {((final_df['word_count'] >= 50) & (final_df['word_count'] < 100)).sum()}")
print(f"100-200 words: {((final_df['word_count'] >= 100) & (final_df['word_count'] < 200)).sum()}")
print(f"200-400 words: {((final_df['word_count'] >= 200) & (final_df['word_count'] < 400)).sum()}")
print(f"> 400 words: {(final_df['word_count'] >= 400).sum()}")




# Ditribution of Word Count before Filtering
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
plt.hist(final_df['word_count'], bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Word Count Distribution (Initial)')
plt.xlim(0, 600)
plt.show()


#%%
#--------------------------------------------------------------------------------------------------------------------
# Filter Data
#--------------------------------------------------------------------------------------------------------------------


final_df = final_df[(final_df['word_count'] >= 50) & (final_df['word_count'] <= 400)]

print("*"*50)
print(f"Dataset Overview(after Filtering)")
print("*"*50)
print(f"Total Observations: {len(final_df)}")
print(f"Dataframe Head: \n{tabulate(final_df.head(2), headers='keys', tablefmt='pretty')}")


# Ditribution of Word Count after Filtering
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
plt.hist(final_df['word_count'], bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Word Count Distribution (After Filtering)')
plt.show()



#--------------------------------------------------------------------------------------------------------------------
# Save Data
#--------------------------------------------------------------------------------------------------------------------
#%%

final_df.to_csv(data_directory + '/Tiny Stories_Cleaned Dataset.csv', index=False)


