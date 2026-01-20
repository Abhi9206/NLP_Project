# evaluate_gnn_slm.py - Evaluation for GNN + Retrieval + GPT-2 model

import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd

from component.gnn_retrieval_slm import GNNRetrievalSLM
from component.retriever import StoryRetriever
from component import utils_gnn
import evaluation_metric as eval_utils

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

CHECKPOINT_PATH = "../Models/GNN_Retrieval_SLM/checkpoints/best_model.pt"
TRAIN_GRAPHS = "graph_data/train_graphs.pt"
VAL_GRAPHS = "graph_data/val_graphs.pt"


SLM_NAME = "gpt2"  # or "openai-community/gpt2"



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------------------------------
# Load graph datasets
# ----------------------------------------------------

print("\nLoading graph datasets...")
train_graphs = torch.load(TRAIN_GRAPHS, weights_only=False)
val_graphs = torch.load(VAL_GRAPHS, weights_only=False)

print(f"Loaded {len(train_graphs):,} train graphs")
print(f"Loaded {len(val_graphs):,} val graphs")

NUM_TEST_STORIES = len(val_graphs)

if NUM_TEST_STORIES:
    val_graphs = val_graphs[:NUM_TEST_STORIES]
    print(f"Testing with {len(val_graphs)} stories")

# ----------------------------------------------------
# Build retriever
# ----------------------------------------------------

print("\nBuilding retrieval index...")
retriever = StoryRetriever(k=3)
train_stories = [g.target_story for g in train_graphs]
retriever.build_index(train_stories)
print("✓ Retrieval index built")

# ----------------------------------------------------
# Reconstruct model architecture
# ----------------------------------------------------

print("\nReconstructing GNN + Retrieval + GPT-2 model...")

model = GNNRetrievalSLM(
    gnn_hidden_dim=256,
    slm_name=SLM_NAME,  # ← GPT-2
    retriever=retriever,
    hf_token=None
).to(device)

# ----------------------------------------------------
# Load checkpoint
# ----------------------------------------------------

print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("✓ Model loaded successfully")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# ----------------------------------------------------
# Generate stories
# ----------------------------------------------------

print("\n" + "=" * 60)
print("GENERATING STORIES")
print("=" * 60)

generated_stories = []
reference_stories = []
keywords_list = []

for i, graph_data in enumerate(val_graphs):
    print(f"\n--- Story {i + 1}/{len(val_graphs)} ---")

    try:
        # Generate story using GNN + Retrieval + GPT-2
        generated_story = utils_gnn.generate_story(
            model,
            graph_data,
            device,
            max_new_tokens=300
        )

        reference_story = graph_data.target_story
        keywords = getattr(graph_data, "keywords", [])

        print(f"Keywords: {keywords}")
        print(f"Generated length: {len(generated_story)} chars")
        print(f"Preview: {generated_story[:100]}...")

        generated_stories.append(generated_story)
        reference_stories.append(reference_story)
        keywords_list.append(keywords)

    except Exception as e:
        print(f"Generation failed: {e}")
        generated_stories.append("")
        reference_stories.append(graph_data.target_story)
        keywords_list.append(getattr(graph_data, "keywords", []))

print(f"\n{'=' * 60}")
print(f"Generation complete: {sum(1 for s in generated_stories if s)} / {len(val_graphs)} successful")
print(f"{'=' * 60}")

# ----------------------------------------------------
# Prepare evaluation table
# ----------------------------------------------------

print("\nPreparing evaluation data...")

eval_df = pd.DataFrame({
    "id": range(len(generated_stories)),
    "story": reference_stories,
    "generated_story": generated_stories,
    "words": keywords_list
})

# Filter valid generations
valid_df = eval_df[eval_df.generated_story.str.len() > 0].copy()

print(f"Valid stories for evaluation: {len(valid_df)} / {len(eval_df)}")

if len(valid_df) == 0:
    print("No valid generations to evaluate!")
    exit(1)

# ----------------------------------------------------
# Run evaluation metrics
# ----------------------------------------------------

print("\n" + "=" * 60)
print("RUNNING EVALUATION METRICS")
print("=" * 60)

try:
    eval_df_with_metrics, summary = eval_utils.evaluate_text_generation(
        valid_df,
        ref_col="story",
        hyp_col="generated_story"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        try:
            print(f"{k:30s}: {v:.4f}")
        except (TypeError, ValueError):
            print(f"{k:30s}: {v}")
    print("=" * 60)

    # Save results
    output_file = f"evaluation_gnn_retrieval_slm_{len(valid_df)}_stories.csv"
    eval_df_with_metrics.to_csv(output_file, index=False)
    print(f"\nSaved detailed results to: {output_file}")

    # # Print sample generations
    # print("\n" + "=" * 60)
    # print("SAMPLE GENERATIONS")
    # print("=" * 60)
    # for i in range(min(1, len(valid_df))):
    #     row = valid_df.iloc[i]
    #     print(f"\nStory {i + 1}:")
    #     print(f"Keywords: {row['words']}")
    #     print(f"Reference (first 150 chars): {row['story'][:150]}...")
    #     print(f"Generated (first 150 chars): {row['generated_story'][:150]}...")
    #     print("-" * 60)

except Exception as e:
    print(f"\nEvaluation failed: {e}")
    import traceback

    traceback.print_exc()

    # Save generation results without metrics
    output_file = f"generation_results_gnn_retrieval_slm_{len(valid_df)}_stories.csv"
    eval_df.to_csv(output_file, index=False)
    print(f"\nSaved generation results (without metrics) to: {output_file}")

print("\nEvaluation script complete!")