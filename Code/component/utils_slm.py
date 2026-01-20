import ast
import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_example(row):
    """Create full training text for Mistral."""
    if isinstance(row["words"], str) and row["words"].startswith("["):
        words_list = ast.literal_eval(row["words"])
    else:
        words_list = row["words"]

    if not isinstance(words_list, list):
        words_list = []

    keywords_str = ", ".join(str(w) for w in words_list)
    prompt_str = str(row["story_beginning_prompt"]).strip()
    story_str = str(row["story"]).strip()

    input_text = (
        "Task: Write a children's story.\n"
        f"Keywords to include: {keywords_str}\n"
        f"Story beginning: {prompt_str}\n"
        f"Complete story:\n{story_str}"
    )
    return input_text


def tokenize_function(tokenizer, max_length):
    """Return tokenization function with closure over tokenizer and max_length."""
    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )
    return _tokenize


def parse_words_column(x):
    """Parse keywords from various formats."""
    if isinstance(x, list):
        return [str(w).strip().lower() for w in x if str(w).strip()]

    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan":
            return []
        try:
            if s.startswith("[") and s.endswith("]"):
                lst = ast.literal_eval(s)
            else:
                lst = s.split(",")
        except Exception:
            lst = []
        return [str(w).strip().lower() for w in lst if str(w).strip()]

    return []


def build_generation_prompt(row):
    """Build prompt without gold story."""
    if isinstance(row["words"], str) and row["words"].startswith("["):
        words_list = ast.literal_eval(row["words"])
    else:
        words_list = row["words"]

    if not isinstance(words_list, list):
        words_list = []

    keywords_str = ", ".join(str(w) for w in words_list)
    prompt_str = str(row["story_beginning_prompt"]).strip()

    prompt = (
        "Task: Write a children's story.\n"
        f"Keywords to include: {keywords_str}\n"
        f"Story beginning: {prompt_str}\n"
        "Complete story:\n"
    )
    return prompt


def generate_story(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15
):
    """Generate story with proper error handling."""
    try:
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            if full_text.startswith(prompt_text):
                generated = full_text[len(prompt_text):].strip()
            else:
                generated = full_text.strip()

            if not generated or len(generated) < 10:
                return "Once upon a time, there was a little story."

            return generated

    except Exception as e:
        print(f"  ❌ Generation error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""