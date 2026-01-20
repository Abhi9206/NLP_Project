#--------------------------------------------------------------------------------------------------------------------
# Import Relevant Packages
#--------------------------------------------------------------------------------------------------------------------
#%%


import torch
import os
import gc
from tqdm import tqdm

from component.utils_gnn import memory_efficient_processing, process_streaming
from component.utils_gnn import StoryGraphBuilder
import pickle

#--------------------------------------------------------------------------------------------------------------------
# Set Working Directory
#--------------------------------------------------------------------------------------------------------------------
#%%

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
token_directory = os.path.join(parent_directory, 'Tokenization')



#--------------------------------------------------------------------------------------------------------------------
# Main Function
#--------------------------------------------------------------------------------------------------------------------
#%%

def main():
    """Main pipeline to build graph dataset"""

    print("\n" + "*" * 60)
    print("Graph Creation")
    print("*" * 60)

    # Initialize builder
    print("\n📦 Initializing StoryGraphBuilder...")
    builder = StoryGraphBuilder(
        spacy_model='en_core_web_sm',
        bert_model='all-MiniLM-L6-v2',
        embedding_dim=384,
        target_nodes=15,
        max_entities=10
    )
    print("✓ Builder initialized!\n")


    USE_MULTIFILE = False  # Set to False for single file

    results = {}

    for split in ['train', 'val', 'test']:
        tokenized_path = os.path.join(token_directory, f'{split}_tokenized.pt')

        if not os.path.exists(tokenized_path):
            print(f"⚠️ File not found: {tokenized_path}")
            continue

        if USE_MULTIFILE:
            # Multi-file strategy
            num_graphs, files = process_streaming(
                builder=builder,
                tokenized_path=tokenized_path,
                split_name=split,
                output_dir='graph_data',
                chunk_size=50000,
                graphs_per_file=500000
            )
            results[split] = (num_graphs, len(files))
        else:
            # Single file strategy
            num_graphs = memory_efficient_processing(
                builder=builder,
                tokenized_path=tokenized_path,
                split_name=split,
                output_dir='graph_data',
                chunk_size=50000
            )
            results[split] = (num_graphs, 1)

        gc.collect()
        print("\n" + "─" * 60 + "\n")

    # Final summary
    print("\n" + "=" * 60)
    print("✅ ALL DATASETS PROCESSED")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        if split in results:
            num_graphs, num_files = results[split]
            print(f"  • {split:5s}: {num_graphs:>8,} graphs in {num_files} file(s)")



if __name__ == "__main__":
    main()