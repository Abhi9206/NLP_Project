# ----------------------------------------------------
# Imports
# ----------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from component import utils_slm
import evaluation_metric as eval_utils

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
CHECKPOINT_DIR = "./story-mistral-best"
VAL_CSV = "../Data/val.csv"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
print(f"Loading base model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading LoRA weights from: {CHECKPOINT_DIR}")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model = model.merge_and_unload()
model.eval()

print(f"Loading tokenizer from: {CHECKPOINT_DIR}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"\nModel loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# ----------------------------------------------------
# Load Validation Data
# ----------------------------------------------------

print(f"\nLoading validation data from: {VAL_CSV}")
val_df = pd.read_csv(VAL_CSV)
print(f"Total validation samples: {len(val_df)}")

NUM_TEST_STORIES = len(val_df)

val_df = val_df.head(NUM_TEST_STORIES).reset_index(drop=True)
print(f"Testing with {len(val_df)} stories")

# ----------------------------------------------------
# Generate Stories
# ----------------------------------------------------
print("\n" + "=" * 60)
print("GENERATING STORIES")
print("=" * 60)

generated_stories = []
failed_count = 0

for i, row in val_df.iterrows():
    print(f"\nStory {i + 1}/{len(val_df)}...")

    try:
        prompt = utils_slm.build_generation_prompt(row)

        gen_story = utils_slm.generate_story(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device
        )

        if gen_story:
            generated_stories.append(gen_story)
        else:
            print(f"Empty generation!")
            generated_stories.append("No story generated.")
            failed_count += 1

    except Exception as e:
        print(f"Failed: {e}")
        generated_stories.append("")
        failed_count += 1

print(f"\n{'=' * 60}")
print(f"Generation complete: {len(val_df) - failed_count}/{len(val_df)} successful")
if failed_count > 0:
    print(f"{failed_count} stories failed to generate")
print(f"{'=' * 60}")

# ----------------------------------------------------
# Prepare Evaluation Data
# ----------------------------------------------------
print("\nPreparing evaluation data...")

val_df["words_clean"] = val_df["words"].apply(utils_slm.parse_words_column)

eval_df = pd.DataFrame({
    "id": range(len(val_df)),
    "story": val_df["story"].astype(str).tolist(),
    "generated_story": generated_stories,
    "words": val_df["words_clean"].tolist(),
})

non_empty_mask = eval_df["generated_story"].str.len() > 0
eval_df_filtered = eval_df[non_empty_mask].copy()


if len(eval_df_filtered) == 0:
    print("No valid generations to evaluate!")
    exit(1)

# ----------------------------------------------------
# Run Evaluation
# ----------------------------------------------------
print("\n" + "=" * 60)
print("RUNNING EVALUATION METRICS")
print("=" * 60)

try:
    eval_df_with_metrics, summary = eval_utils.evaluate_text_generation(
        eval_df_filtered,
        ref_col="story",
        hyp_col="generated_story",
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        try:
            print(f"{k:30s}: {v:.4f}")
        except (TypeError, ValueError):
            print(f"{k:30s}: {v}")
    print("=" * 60)

    output_file = f"evaluation_test_{NUM_TEST_STORIES}_stories.csv"
    eval_df_with_metrics.to_csv(output_file, index=False)
    print(f"\nSaved detailed results to: {output_file}")



except Exception as e:
    print(f"\nEvaluation failed: {e}")
    import traceback
    traceback.print_exc()

    output_file = f"generation_results_{NUM_TEST_STORIES}_stories.csv"
    eval_df.to_csv(output_file, index=False)
    print(f"\nSaved generation results (without metrics) to: {output_file}")

print("\nEvaluation script complete!")